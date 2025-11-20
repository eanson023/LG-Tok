from os.path import join as pjoin

import torch
import torch.optim as optim
import torch.nn.utils
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

import time
import os
from collections import OrderedDict, defaultdict
from utils.eval_t2m import evaluation_vqvae
from utils.utils import print_current_loss


def def_value():
    return 0.0

class VQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device
        self.scaler = GradScaler()

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()
            

    # enable autocast for mixed precision
    @torch.cuda.amp.autocast()
    def forward(self, batch_data):
        if isinstance(batch_data, tuple):
            batch_data1, batch_data2 = batch_data
            motions1, m_lens1, texts1 = batch_data1
            motions2, m_lens2, texts2 = batch_data2
            motions = torch.cat([motions1, motions2], dim=0).detach().to(self.device).float()
            m_lens = torch.cat([m_lens1, m_lens2], dim=0)
            texts = texts1 + texts2
            idx = torch.randperm(len(motions))
            motions = motions[idx]
            m_lens = m_lens[idx]
            texts = [texts[i] for i in idx]
        else:
            motions, m_lens, texts = batch_data
            motions = motions.detach().to(self.device).float()

        motions = motions[:, :max(m_lens)]

        pred_motion, code_usage, loss_commit, perplexity, _ = self.vq_model(motions, m_lens, texts)
        
        self.motions = motions # (bs, T, dim)
        self.pred_motion = pred_motion
        
        loss_rec = self.l1_criterion(pred_motion, motions)

        pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        loss_explicit = self.l1_criterion(pred_local_pos, local_pos)

        loss = loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * loss_commit

        return loss, loss_rec, loss_explicit, loss_commit, code_usage, perplexity


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.vq_model.load_state_dict(checkpoint['vq_model'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('text_model.') for k in missing_keys])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint["scaler"])
        
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_loader, train_loader_ood, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d" % (epoch, it))
        
        if self.opt.is_continue:
            milestones = [int(len(train_loader) * m) - it for m in self.opt.milestones if int(len(train_loader) * m) > it]
            milestones = [max(1, ms) for ms in milestones]  
        else:
            milestones = [int(len(train_loader) * m) for m in self.opt.milestones]

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=milestones, gamma=self.opt.gamma)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(eval_val_loader)))

        logs = defaultdict(def_value, OrderedDict())
            
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_vqvae(
            self.opt.model_dir, eval_val_loader,  self.vq_model, self.logger, epoch, best_fid=1000,
            best_div=100, best_top1=0,
            best_top2=0, best_top3=0, best_matching=100,
            eval_wrapper=eval_wrapper, save=False)

        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            if train_loader_ood is not None:
                data_loader = zip(train_loader, train_loader_ood)
            else:
                data_loader = train_loader
            for i, batch_data in enumerate(data_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                loss, loss_rec, loss_vel, loss_commit, code_usage, perplexity = self.forward(batch_data)
            
                self.opt_vq_model.zero_grad()
                
                self.scaler.scale(loss).backward()
                
                if self.opt.grad_clip > 0:
                    self.scaler.unscale_(self.opt_vq_model)
                    torch.nn.utils.clip_grad_norm_(self.vq_model.parameters(), self.opt.grad_clip)
                
                self.scaler.step(self.opt_vq_model)
                self.scaler.update()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()

                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['code_usage'] += code_usage.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            self.val(val_loader, epoch)
            if epoch % self.opt.eval_every_e == 0:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching = self.eval(eval_val_loader, eval_wrapper, plot_eval, epoch,
                                                                                              best_fid, best_div, best_top1, best_top2, best_top3, best_matching)

        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

    
    def val(self, val_loader, epoch):
        print('Validation time:')
        self.vq_model.eval()
        val_loss_rec = []
        val_loss_vel = []
        val_loss_commit = []
        val_loss = []
        val_code_usage = []
        val_perpexity = []
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                loss, loss_rec, loss_vel, loss_commit, code_usage, perplexity = self.forward(batch_data)
                val_loss.append(loss.item())
                val_loss_rec.append(loss_rec.item())
                val_loss_vel.append(loss_vel.item())
                val_loss_commit.append(loss_commit.item())
                val_code_usage.append(code_usage.item())
                val_perpexity.append(perplexity.item())

        self.logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
        self.logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
        self.logger.add_scalar('Val/loss_vel', sum(val_loss_vel) / len(val_loss_vel), epoch)
        self.logger.add_scalar('Val/loss_commit', sum(val_loss_commit) / len(val_loss), epoch)
        self.logger.add_scalar('Val/code_usage', sum(val_loss_commit) / len(val_loss), epoch)
        self.logger.add_scalar('Val/perplexity', sum(val_perpexity) / len(val_loss_rec), epoch)

        print('Validation Loss: %.5f Reconstruction: %.5f, Velocity: %.5f, Commit: %.5f' %
                (sum(val_loss)/len(val_loss), sum(val_loss_rec)/len(val_loss), 
                sum(val_loss_vel)/len(val_loss), sum(val_loss_commit)/len(val_loss)))

    
    def eval(self, eval_val_loader, eval_wrapper, plot_eval, epoch, best_fid, best_div, best_top1, best_top2, best_top3, best_matching):
        self.vq_model.eval()

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_vqvae(
        self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
        best_div=best_div, best_top1=best_top1,
        best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper)

        data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0).detach().cpu().numpy()
        # np.save(pjoin(self.opt.eval_dir, 'E%04d.npy' % (epoch)), data)
        if plot_eval is not None:
            save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
            os.makedirs(save_dir, exist_ok=True)
            plot_eval(data, save_dir)
        
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching


