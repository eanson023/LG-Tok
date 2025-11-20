from options.base_option import BaseOptions
import argparse

class TrainT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epoch for training')
        # self.parser.add_argument('--max_iters', type=int, default=150_000, help='Training iterations')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate schedule factor')
        self.parser.add_argument('--milestones', default=[300], nargs="+", type=int, help="learning rate schedule (opech)")
        self.parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')

        '''Condition'''
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument("--seed", default=2025, type=int, help="Seed")

        self.parser.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress, (iteration)')
        # self.parser.add_argument('--save_every_e', type=int, default=100, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=1, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--eval_start_e', type=int, default=100, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving checkpoint, (iteration)')

        self.parser.add_argument('--pkeep', type=float, default=0.5, help='keep rate for gpt training')
        self.parser.add_argument('--tiny', action="store_true", help="training on small datasets, small model")
        self.parser.add_argument("--tfg", type=float, default=1.0, help="Text-free guidance scale for tokenizer")

        # special hyper paramaters for MMM
        self.parser.add_argument("--num_local_layer", type=int, default=2, help='number of cross attention layer')

        #special hyper paramaters for MoMask
        self.parser.add_argument("--stage", type=int, default=1, choices=[1, 2], help="momask's training stage")
        self.parser.add_argument('--share_weight', action="store_true", help='Whether to share weight for projection/embedding, for residual transformer.')

        self.is_train = True