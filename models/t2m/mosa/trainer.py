import torch
from collections import defaultdict
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_transformer


def def_value():
    return 0.0

class TransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        self.epoch = 0

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_len = batch_data
        motion = motion.detach().float().to(self.device)
        m_len = m_len.detach().long().to(self.device)

        # (b, n, q)
        x = self.vq_model.quantize(motion, m_len, conds)
        pkeep = self.opt.pkeep
        new_x = []

        if self.t2m_transformer.training:
            for q, X_q in enumerate(x):
                if pkeep == -1:
                    proba = np.random.rand(1)[0]
                    mask = torch.bernoulli(proba * torch.ones(X_q.shape,
                                                                    device=X_q.device))
                else:
                    mask = torch.bernoulli(pkeep * torch.ones(X_q.shape,
                                                                    device=X_q.device))
                mask = mask.round().to(dtype=torch.int64)
                rest_X_q = torch.randint_like(X_q, self.vq_model.quantizer.nb_codes[q])
                new_X= mask*X_q + (1-mask)*rest_X_q
                new_x.append(new_X)
        else:
            new_x = x
        x_inputs = self.vq_model.quantizer.idxBT_to_t2m_input(new_x)
        # x = x[:start_drop_quantize_index+1]

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        _loss, _pred_ids, _acc, _acc_scales = self.t2m_transformer(x_inputs, conds, x, m_len, p_drop_factor=(1-self.epoch/self.opt.max_epoch))

        return _loss, _acc, _acc_scales

    def update(self, batch_data):
        loss, acc, acc_scales = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss.item(), acc, acc_scales

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=[int(len(train_loader) * m) for m in self.opt.milestones],
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_clip_score, writer = evaluation_transformer(
            self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, best_clip_score=0, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=False,
            train_mean=train_loader.dataset.mean, train_std=train_loader.dataset.std
        )
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            self.t2m_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc, acc_scales = self.update(batch_data=batch)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']
                for k, acc_s in enumerate(acc_scales):
                    logs[f'acc_scale_{str(k)}'] += acc_s

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_acc = []
            val_acc_scales = defaultdict(list)
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc, acc_scales = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)
                    for k, acc_s in enumerate(acc_scales):
                        val_acc_scales[k].append(acc_s)

            self.epoch = epoch

            print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)
            # Compute and log mean validation accuracy for each patch_num
            mean_val_acc_scales = {q: np.mean(val_acc_scales[q]) for q in val_acc_scales}
            for q, mean_acc_s in mean_val_acc_scales.items():
                self.logger.add_scalar(f'Val/acc_patch_{q}', mean_acc_s, epoch)

            # if np.mean(val_acc) > best_acc:
            #     print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
            #     self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
            #     best_acc = np.mean(val_acc)

            if epoch % self.opt.eval_every_e == 0:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_clip_score, writer = evaluation_transformer(
                    self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, best_clip_score=best_clip_score, eval_wrapper=eval_wrapper,
                    plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0), train_mean=train_loader.dataset.mean, train_std=train_loader.dataset.std
                )
