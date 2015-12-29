

# RLNTM is licensed under a
# Creative Commons Attribution-NonCommercial 4.0 International License.

# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.


"""
Start a new process with an experiment.  Convenient for grid and random searches.
"""
def run(name,
        write_path="NULL",
        rnn_size='50',
        cmd='up',
        init_scale='0.1',
        vis_freq='50',
        LR='0.025',
        momentum='.9',
        batch_size='300',
        entropy_gain_factor='0.2',
        entropy_gain_factor_iter='30000',
        curriculum_error_rate_thresh='.2',
        seq_len='14',
        T='33',
        action_slowdown='.03',
        save_freq='50',
        cpu='8',
        priority='100',
        task_name='repeatCopy',
        seed='12345',
        num_iter='3e4',
        force_input_forward='0',
        direct_mem_input='0',
        direct_input_input='0',
        super_direct='0',
        mem_size='-1',
        length_testing='0',
        T_of_length_testing='0',
        seq_len_of_length_testing='0',
        extra='',

        depth='1',
        baseline_slowdown = "0.05",
        efficient_mem='1',
        max_grad_norm_rl='2',
        max_grad_norm_baseline='5',
        print_freq='20',
        forget_gate_bias='0',
        per_step_gain='0',
        separate_baseline='1',
        curriculum_decay = ".8",
        mem_write_range='2',
        mem_read_range='2',
        input_read_range='2',
        curriculum_refractory_period = "100"):

  exe = 'train_curriculum0.lua'

  args = [
      " rlntm_pkgs/%s" % exe,
      "-name %s" % name,
      "-seed %s" % seed,
      "-T %s" % T,
      "-seq_len %s" % seq_len, 
      "-init_scale %s" % init_scale,
      "-rnn_size %s" % rnn_size,
      "-depth %s" % depth,
      "-action_slowdown %s" % action_slowdown,
      "-baseline_slowdown %s" % baseline_slowdown,
      "-efficient_mem %s" % efficient_mem,
      "-LR %s" % LR,
      "-momentum %s" % momentum,
      "-max_grad_norm_rl %s" % max_grad_norm_rl,
      "-max_grad_norm_baseline %s" % max_grad_norm_baseline,
      "-batch_size %s" % batch_size,
      "-num_iter %s" % num_iter,
      "-print_freq %s" % print_freq,
      "-vis_freq %s" % vis_freq,
      "-forget_gate_bias %s" % forget_gate_bias,
      "-per_step_gain %s" % per_step_gain,
      "-entropy_gain_factor %s" % entropy_gain_factor,
      "-entropy_gain_factor_iter %s" % entropy_gain_factor_iter,
      "-separate_baseline %s" % separate_baseline,
      "-curriculum_error_rate_thresh %s" % curriculum_error_rate_thresh,
      "-curriculum_decay %s" % curriculum_decay,
      "-task_name %s" % task_name,
      "-mem_write_range %s" % mem_write_range,
      "-mem_read_range %s" % mem_read_range,
      "-input_read_range %s" % input_read_range,
      "-write_path %s" % write_path,
      "-direct_mem_input %s" % direct_mem_input,
      "-direct_input_input %s" % direct_mem_input,
      "-curriculum_refractory_period %s" % curriculum_refractory_period,
      "-force_input_forward %s" % force_input_forward,
      "-super_direct %s" % super_direct,
      "-mem_size %s" % mem_size,
      "-length_testing %s" % length_testing,
      "-T_of_length_testing %s" % T_of_length_testing,
      "-seq_len_of_length_testing %s" % seq_len_of_length_testing
    ]

  cmd = ' '.join(args)

  from multiprocessing import Process
  def run():
    import os
    print cmd
    os.system('/usr/bin/torch %s' % cmd)


  p = Process(target=run)
  p.start()
  return cmd, p
