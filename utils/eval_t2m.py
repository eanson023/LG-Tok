import os

import clip
import numpy as np
import torch
# from scipy import linalg
from utils.metrics import *
import torch.nn.functional as F
# import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric
from utils.tools import lengths_to_mask
from tqdm import tqdm
#
#
# def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
#     xyz = xyz[:1]
#     bs, seq = xyz.shape[:2]
#     xyz = xyz.reshape(bs, seq, -1, 3)
#     plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
#     plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
#     writer.add_video(tag, plot_xyz, nb_iter, fps=20)


@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, writer, it, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        (et, em), (_, _) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, motion.clone(), m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        pred_pose_eval, *_ = net(motion, m_length, caption)

        (et_pred, em_pred), (_, _) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Iter %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (it, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, it)
        writer.add_scalar('./Test/Diversity', diversity, it)
        writer.add_scalar('./Test/top1', R_precision[0], it)
        writer.add_scalar('./Test/top2', R_precision[1], it)
        writer.add_scalar('./Test/top3', R_precision[2], it)
        writer.add_scalar('./Test/matching_score', matching_score_pred, it)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'total_it': it}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'total_it': it}, os.path.join(out_dir, 'net_best_mm.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe_cbusage(val_loader, net, repeat_id, eval_wrapper, num_joint, tfg, mfg, train_mean=None, train_std=None, ):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    codebook_usages = []
    if hasattr(net.quantizer, 'nb_codes'):
        codebook_usages = [torch.zeros(nb_code, device=next(net.parameters()).device) for nb_code in net.quantizer.nb_codes]

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        (et, em), (_, _) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, motion.clone(), m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        motion = val_loader.dataset.transform(bgt, train_mean, train_std)
        motion = torch.from_numpy(motion).cuda()

        # pred_pose_eval = motion
        pred_pose_eval, *_, indices = net(motion, m_length, caption)

        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy(), train_mean, train_std)
        bpredd = val_loader.dataset.transform(bpred)

        (et_pred, em_pred), (_, _) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, caption, \
                                                                    torch.from_numpy(bpredd).to(motion.device), m_length)
        for c in range(len(codebook_usages)):
            unique_indices = torch.unique(indices[c])
            codebook_usages[c][unique_indices] += 1

        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
    cb_usage = torch.tensor(0.)
    if hasattr(net.quantizer, 'nb_codes'):
        cb_usage = sum([(codebook_usage>0).sum() / codebook_usage.shape[0] for codebook_usage in codebook_usages])/len(codebook_usages)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f CB Usage. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe, cb_usage)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, mpjpe, cb_usage


@torch.no_grad()
def evaluation_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, best_clip_score, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, train_mean=None, train_std=None, model="mosa"):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    clip_score_real = 0
    clip_score_gt = 0
    if "kit" in out_dir:
        cond_scale = 2
    elif "motionx" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        pose = pose.cuda().float()

        # (b, seqlen)
        if model == "mosa":
            pred_latents = trans.generate(clip_text, m_length, cond_scale, vq_model, temperature=1)
        elif model == "mar":
            pred_latents = trans.generate(clip_text, vq_model.num_latent_tokens, 18, cond_scale, temperature=1, hard_pseudo_reorder=False)
        elif model == "momask_stage1":
            mids = trans.generate(clip_text, vq_model.num_latent_tokens, 18, cond_scale, temperature=1)
            # In fact, momask's pred_latents is mids
            pred_latents = mids.unsqueeze_(-1)
        elif model == "momask_stage2":
            code_indices = vq_model.quantize(pose, m_length, clip_text)
            # In fact, momask's pred_latents is mids
            pred_latents = trans.generate(code_indices[..., 0], clip_text, vq_model.num_latent_tokens, temperature=1, cond_scale=cond_scale)

        # decode
        pred_motions = vq_model.dequantize(pred_latents, m_length, clip_text, tfg=0)

        pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy(), train_mean, train_std)
        pred_motions = torch.from_numpy(val_loader.dataset.transform(pred_motions)).to(m_length.device)

        (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, 
                                                          clip_text,
                                                          pred_motions,
                                                          m_length)


        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text, pose, m_length)

        batch_clip_score_pred = 0
        batch_clip_score = 0
        for j in range(bs):
            single_em = em_pred_clip[j]
            single_et = et_pred_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score_pred += clip_score

            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score

        clip_score_real += batch_clip_score_pred
        clip_score_gt += batch_clip_score

        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred} clip score real. {clip_score_gt} clip score. {clip_score_real}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)
    writer.add_scalar('./Test/top1', R_precision[0], ep)
    writer.add_scalar('./Test/top2', R_precision[1], ep)
    writer.add_scalar('./Test/top3', R_precision[2], ep)
    writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
    writer.add_scalar('./Test/clip_score', clip_score_real, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_mm.tar'), ep)

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if clip_score_real > best_clip_score:
        msg = f"--> --> \t CLIP-score Improved from {best_clip_score:.4f} to {clip_score_real:.4f} !!!"
        print(msg)
        best_clip_score = clip_score_real

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)
    

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_clip_score, writer


@torch.no_grad()
def evaluation_transformer_test(val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                cond_scale, tfg, top_k, top_p, temperature, force_mask=False, cal_mm=True, train_mean=None, train_std=None, model="mosa"):
    if model == "momask":
        trans, res_trans = trans
        res_trans.eval()
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    clip_score_gt = 0
    clip_score_pred = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        print('-----------------', i, '-----------------')
        # print(i)
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        pose = pose.cuda().float()

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                # print('-----------------', _, '-----------------')
                if model == "mosa":
                    pred_latents = trans.generate(clip_text, m_length, cond_scale, vq_model, top_k=top_k, top_p=top_p, temperature=temperature)
                elif model == "mar":
                    pred_latents = trans.generate(clip_text, vq_model.num_latent_tokens, 18, cond_scale, temperature=1, hard_pseudo_reorder=False)
                elif model == "momask":
                    mids = trans.generate(clip_text, vq_model.num_latent_tokens, 10, cond_scale,
                                      temperature=temperature, topk_filter_thres=0.9,
                                      gsample=True, force_mask=force_mask)
                    # In fact, momask's pred_latents is mids
                    # pred_latents = mids.unsqueeze_(-1)
                    pred_latents = res_trans.generate(mids, clip_text, vq_model.num_latent_tokens, temperature=1, cond_scale=5)
                pred_motions = vq_model.dequantize(pred_latents, m_lens=m_length, texts=clip_text, tfg=tfg)

                pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy(), train_mean, train_std)
                pred_motions = torch.from_numpy(val_loader.dataset.transform(pred_motions)).to(m_length.device)

                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            if model == "mosa":
                pred_latents = trans.generate(clip_text, m_length, cond_scale, vq_model, top_k=top_k, top_p=top_p, temperature=temperature)
            elif model == "mar":
                pred_latents = trans.generate(clip_text, vq_model.num_latent_tokens, 18, cond_scale, temperature=1, hard_pseudo_reorder=False)
            elif model == "momask":
                mids = trans.generate(clip_text, vq_model.num_latent_tokens, 10, cond_scale,
                                    temperature=temperature, topk_filter_thres=0.9,
                                    gsample=True, force_mask=force_mask)
                # pred_latents = mids.unsqueeze_(-1)
                # In fact, momask's pred_latents is mids
                pred_latents = res_trans.generate(mids, clip_text, vq_model.num_latent_tokens, temperature=1, cond_scale=5)
            pred_motions = vq_model.dequantize(pred_latents, m_lens=m_length, texts=clip_text, tfg=tfg)

            pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy(), train_mean, train_std)
            pred_motions = torch.from_numpy(val_loader.dataset.transform(pred_motions)).to(m_length.device)
            
            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text, pred_motions.clone(),
                                                                  m_length)

        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, clip_text, pose, m_length)

        batch_clip_score_pred = 0
        batch_clip_score = 0
        for j in range(bs):
            single_em = em_pred_clip[j]
            single_et = et_pred_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score_pred += clip_score

            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score

        clip_score_gt += batch_clip_score
        clip_score_pred += batch_clip_score_pred

        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    clip_score_gt = clip_score_gt / nb_sample
    clip_score_pred = clip_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}, " \
          f"clip_score_real. {clip_score_gt:.4f}, clip_score_pred. {clip_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, clip_score_pred, multimodality


@torch.no_grad()
def evaluation_transformer_test_edit(task, val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                cond_scale, top_k, top_p, temperature, force_mask=False, cal_mm=True):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    def create_edit_mask(m_len, task):
        edit_mask = torch.zeros(m_len)
        if task == 'inpainting':
            start = int(m_len * 0.25)
            end = int(m_len * 0.75)
            edit_mask[start:end] = 1
        elif task == 'outpainting':
            start = int(m_len * 0.25)
            end = int(m_len * 0.75)
            edit_mask[:start] = 1
            edit_mask[end:] = 1
        elif task == 'suffix':
            end = int(m_len * 0.5)
            edit_mask[:end] = 1
        elif task == 'prefix':
            start = int(m_len * 0.5)
            edit_mask[start:] = 1
        return edit_mask


    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        # print(i)
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()
        pose = pose.cuda()
        
        edit_masks = lengths_to_mask(m_length//4, max(m_length//4))
        for j, m_len in enumerate(m_length//4):
            edit_masks[j, :m_len] = create_edit_mask(m_len, task)
        edit_masks = edit_masks.float()
        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                tokens = vq_model.quantize(pose, m_length)
                pred_quantized = trans.edit(clip_text, 
                                    tokens, 
                                    edit_masks,
                                    m_length,
                                    cond_scale,
                                    vq_model,
                                    top_k, 
                                    top_p, 
                                    temperature=temperature,
                                    more_smooth=False)
                pred_motions = vq_model.forward_decoder(pred_quantized)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            tokens = vq_model.quantize(pose, m_length)
            pred_quantized = trans.edit(clip_text, 
                                tokens, 
                                edit_masks,
                                m_length,
                                cond_scale,
                                vq_model,
                                top_k, 
                                top_p, 
                                temperature=temperature,
                                more_smooth=False)
            pred_motions = vq_model.forward_decoder(pred_quantized)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality

