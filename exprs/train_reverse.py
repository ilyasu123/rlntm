

# RLNTM is licensed under a
# Creative Commons Attribution-NonCommercial 4.0 International License.

# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.


from runs_utils_new import run
import inspect

name = 'train_reverse-1'

run(name,
    write_path=name,
    cmd='up',
    LR='0.05',
    rnn_size ='25',
    T='30',
    seq_len='30',
    momentum='.9',
    entropy_gain_factor='0',
    curriculum_error_rate_thresh='.25',
    action_slowdown='0.01',
    entropy_gain_factor_iter = '50000',
    num_iter='50000',
    task_name='revCurriculum',
    seed='3',
    force_input_forward='1',
    super_direct='1')
                          
     
