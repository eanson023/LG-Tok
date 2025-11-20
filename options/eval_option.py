from options.base_option import BaseOptions

class EvalT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default="latest", help='Checkpoint you want to use, {latest, net_best_fid, etc}')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

        self.parser.add_argument('--ext', type=str, default='text2motion', help='Extension of the result file or folder')
        self.parser.add_argument("--num_batch", default=2, type=int,
                                 help="Number of batch for generation")
        self.parser.add_argument("--repeat_times", default=1, type=int,
                                 help="Number of repetitions, per sample text prompt")
        self.parser.add_argument("--cond_scale", default=4, type=float,
                                 help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
        self.parser.add_argument("--temperature", default=1., type=float,
                                 help="Sampling Temperature.")
        self.parser.add_argument("--top_p", default=0.95, type=float,
                                 help="Filter out percentil low prop entries.")
        self.parser.add_argument("--top_k", default=0.8, type=float,
                                 help="Filter out percentil low prop entries.")
        self.parser.add_argument("--tfg", type=float, default=1.0, help="Text-free guidance scale")
        self.parser.add_argument("--seed", default=2025, type=int)

        # self.parser.add_argument('--est_length', action="store_true", help='Training iterations')

        self.parser.add_argument('--text_path', type=str, default="", help='Text prompt file')

        self.parser.add_argument('-msec', '--mask_edit_section', nargs='*', type=str, help='Indicate sections for editing, use comma to separate the start and end of a section'
                                 'type int will specify the token frame, type float will specify the ratio of seq_len')
        self.parser.add_argument('--text_prompt', default='', type=str, help="A text prompt to be generated. If empty, will take text prompts from dataset.")
        self.parser.add_argument('--source_motion', default='example_data/000612.npy', type=str, help="Source motion path for editing. (new_joint_vecs format .npy file)")
        self.parser.add_argument("--motion_length", default=196, type=int,
                                 help="Motion length for generation, only applicable with single text prompt.")
        
        # special hyper-parameters for MoMask
        self.parser.add_argument('--res_name', type=str, help='Residual Transformer Name for Momask')
        
        self.is_train = False
