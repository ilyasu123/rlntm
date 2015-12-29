

# RLNTM is licensed under a
# Creative Commons Attribution-NonCommercial 4.0 International License.

# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.


from runs_utils_new import run


name = 'train_repeatCopy-2'

run(name,
    write_path=name,
    cmd='up',
    LR='0.05',
    rnn_size ='25',
    T='44',
    seq_len='9',
    momentum='.9',
    entropy_gain_factor='0',
    curriculum_error_rate_thresh='.2',
    action_slowdown='0.03',
    entropy_gain_factor_iter = '50000',
    num_iter='50000',
    task_name='repeatCopyN',
    seed='10',
    force_input_forward='0',
    super_direct='1')
                          
     
