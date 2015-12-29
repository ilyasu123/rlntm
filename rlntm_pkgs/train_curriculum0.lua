
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

package.path = package.path .. ';rlntm_pkgs/?.lua'
CHECK_GRAD = false -- global variable

print ('num threads = ',torch.getnumthreads())
torch.setnumthreads(8) 
print ('what about now:  num threads = ',torch.getnumthreads())
print ('OK. Number of threads is good. ')


--- check-pointing.  OK, I accept it. 
local misc = require 'misc'
local stringx = require 'pl.stringx'

require 'strict' -- no more global variables after strict is defined. 
local rlntm = require 'rlntm'
local tasks = require 'tasks'
local file = require 'file'
local efficient_mem = require 'efficient_mem'
local visualize = require 'visualize'
require 'optim'
---require 'optimx'  <-- I couldn't install optimx, but it's OK since we don't need RMSprop
local optims = {sgd = optim.sgd, false} --rmsprop = optimx.rmsprop}


require 'hack'
mode = 'hack' -- global variable

---- basically, save stdout. 

local cmd = torch.CmdLine()
cmd:text('RL:NTM!')
cmd:option('-name', 'NULL')
cmd:option('-seed', 1)
cmd:option('-T', 25)
cmd:option('-seq_len', 25)
cmd:option('-init_scale', 0.1)
cmd:option('-rnn_size', 50)
cmd:option('-depth', 1)
cmd:option('-action_slowdown', .1)
cmd:option('-baseline_slowdown', 0.05)
cmd:option('-efficient_mem', 1)
cmd:option('-useRMSProp', 0)
cmd:option('-LR', 0.25)
cmd:option('-momentum', 0.9)
cmd:option('-max_grad_norm_rl', 2)
cmd:option('-max_grad_norm_baseline', 5)
cmd:option('-batch_size', 200)
cmd:option('-num_iter', 1e6)
cmd:option('-print_freq', 20)
cmd:option('-vis_freq', 40)
cmd:option('-forget_gate_bias', 0)
cmd:option('-per_step_gain', -0.0)
cmd:option('-entropy_gain_factor', 0.2)
cmd:option('-entropy_gain_factor_iter', 30000)
cmd:option('-separate_baseline', 1)
cmd:option('-curriculum_error_rate_thresh', 0.25)
cmd:option('-curriculum_decay', .8)
cmd:option('-task_name', 'revCurriculum')
cmd:option('-mem_write_range', 2)
cmd:option('-mem_read_range', 2)
cmd:option('-input_read_range', 2)
cmd:option('-write_path', 'expr2')
cmd:option('-direct_mem_input', 0)
cmd:option('-direct_input_input', 0)
cmd:option('-curriculum_refractory_period', 100)
cmd:option('-save_freq', 50)
cmd:option('-force_input_forward', 0)
cmd:option('-super_direct', 0)
cmd:option('-mem_size', -1)
cmd:option('-length_testing', 0)
cmd:option('-T_of_length_testing', 0)
cmd:option('-seq_len_of_length_testing', 0)
cmd:option('-task_visualize', 0)

local num_symbols = #visualize.char_map




cmd:text()
local params = cmd:parse(arg)
print ('params:')

torch.manualSeed(params.seed)

if params.super_direct == 1 then params.emb_size = num_symbols end
if params.mem_size == -1 and params.super_direct == 1 then params.mem_size = params.emb_size end
if params.mem_size == -1 and params.super_direct == 0 then params.mem_size = params.rnn_size end
print (params)

local length_testing = params.length_testing
local T_of_length_testing = params.T_of_length_testing
local seq_len_of_length_testing = params.seq_len_of_length_testing
local task_visualize = params.task_visualize
assert (task_visualize == 0 or task_visualize == 1)
params.length_testing = nil --- remove length_testing from params since it shouldn't affect the name.
params.T_of_length_testing = nil
params.seq_len_of_length_testing = nil
params.task_visualize = nil

local root_path = 'rlntm_runs/'
local prefix_path = string.format('%s/%s/', root_path, params.write_path)
params.write_path = nil  --- this is how we delete

if not file.Exists(prefix_path) then
   file.MakeDir(prefix_path)
   print (string.format('the path %s does not exist, so we created it.', prefix_path))
else
   print (string.format('the path %s already exists!', prefix_path))
end

local parts = {}
for i,j in pairs(params) do
   if j == nil then 
      print (string.format("darn it, params: there is an (%s, %s) pair with a NIL! :(", i, j))
   end
   if j ~= nil then 
      table.insert(parts, string.format('%s_%s', i, j))
   end
end
----- For some reason, I am unable to store very long run names.  Thus, we will store
----- only the high level name of the run only:
----- local run_sig = stringx.join('__', parts)
local run_sig = params.name
local fname = prefix_path .. run_sig
local fname_txt = prefix_path .. run_sig .. '.txt'
local fname_stdout = prefix_path .. run_sig .. '.stdout'
print ('fname = ', fname)
print ('fname_txt = ', fname_txt)
print ('fname_stdout = ', fname_stdout)

local check_point_exists = file.Exists(fname)

--logging.info('Options')
--logging.info(params)
--logging.info('Launching the experiment')

assert (tasks[params.task_name] ~= nil)
assert (tasks[params.task_name .. '_curriculum'] ~= nil)

local curriculum
if length_testing == 0 then
   curriculum = tasks[params.task_name .. '_curriculum']
elseif length_testing == 1 then
   assert (tasks[params.task_name .. '_hardcurriculum'] ~= nil)
   curriculum = tasks[params.task_name .. '_hardcurriculum']
else
   error ("length_testing should be 0 or 1 only")
end

--- step 1:  instantiate the network
local T = params.T

local rnn_size = params.rnn_size
local emb_size = num_symbols
local num_out_symbols = num_symbols
local depth = params.depth
local mem_tape_len = 30
local action_slowdown = params.action_slowdown
local baseline_slowdown = params.baseline_slowdown
local batch_size = params.batch_size
local forget_gate_bias = params.forget_gate_bias
local per_step_gain = params.per_step_gain
local entropy_gain_factor = params.entropy_gain_factor
local LR = params.LR
local separate_baseline = params.separate_baseline
local momentum = params.momentum
local in_moves_table  = {-1, 0, 1}
local mem_moves_table = {-1, 0, 1}
local out_moves_table = {0, 1}
local seq_len = params.seq_len
local num_iter = params.num_iter
local vis_freq = params.vis_freq
local optim_func
if params.useRMSProp == 1 then
    optim_func = optims.rmsprop
else
    optim_func = optims.sgd
end


if length_testing == 1 then
   T = T_of_length_testing
   seq_len = seq_len_of_length_testing  
   batch_size = 2
   LR = 0
   print ('length_testing:batch_size=1')
   print ('length_testing:T=',T)
   print ('length_testing:seq_len=',T)
end

if task_visualize == 1 then
   vis_freq = 1
end


-------- step 2:  allocate various parameter objects. 
local max_in_len = seq_len
local special_init = true
print ('RLNTM:creating...')
local r, set_gain_factor
   = rlntm.create(T, 
		  max_in_len, 
		  rnn_size, 
		  emb_size, 
		  depth, 
		  num_out_symbols, 
		  in_moves_table, 
		  mem_moves_table, 
		  out_moves_table, 
		  action_slowdown, 
		  baseline_slowdown, 
		  forget_gate_bias, 
		  per_step_gain, 
		  entropy_gain_factor, 
		  separate_baseline, 
		  special_init, 
		  params.mem_write_range, 
		  params.mem_read_range, 
		  params.input_read_range,
		  params.direct_mem_input,
		  params.direct_input_input,
		  params.force_input_forward,
		  params.super_direct,
		  params.mem_size,
		  length_testing
)
print ('RLNTM: created!')

local x, dx = r:getParameters()





--- at this point, x has "1"s on all the RL values. 
local x_rl = x:clone()
local x_baseline = x_rl:clone():mul(-1):add(1)
print ('x_rl:min() = ', x_rl:min())
print ('x_rl:max() = ', x_rl:max())
assert (x_rl:min() == 0 and x_rl:max() == 1)


local v = torch.zeros(x:size())

local init_H = {}
for d = 1, 2*depth do
    init_H[d] = torch.zeros(batch_size, rnn_size):fill(0.1)
end






------------- the optimization state
local optim_state = {}
optim_state.learningRate = LR
optim_state.momentum = momentum




-- local function concatTensor(...)
--    local tensors = {...}
--    local s = 0
--    for _,t in pairs(tensors) do
--       s = s + t:nElement()
--    end
--    local ans = torch.zeros(s)
--    local p = 0
--    for _,t in pairs(tensors) do
--       ans[{{p + 1, p + t:nElement()}}]:copy(t) 
--       p = p + t:nElement()
--    end
--    return ans
-- end

-- local function unconcatTensor(x, tensors)
--    local p = 0
--    for _,t in pairs(tensors) do
--       t:copy(x[{{p + 1, p + t:nElement()}}])
--       p = p + t:nElement()
--    end
-- end

local input_mem_params = torch.zeros(1, 1, params.mem_size):expand(batch_size, mem_tape_len, params.mem_size):clone()

local acc_loss, acc_batch_size = 0, 0
local input_mem_2 = torch.zeros(1)
local curriculum_refractory_period = params.curriculum_refractory_period

local cur_curriculum_error_rate 
local last_curriculum_advance 
local curriculum_level 

---- BEGIN TRAINING.  TRAINING IS GOOD.
local check_point = {}
local startStep
local extraTime 
local total_stdout
print ('about to load checkpoint!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
if  check_point_exists == true and params.name ~= 'NULL' then
   print ('a check point EXISTS! hooray!  Loading it!')
   local contents = file.GetContents(fname)
   print ('contents == nil =', contents == nil)
   print (string.format('type of contents = %s', type(contents)))
   check_point = torch.deserialize(contents)
   x:copy(check_point.x)
   startStep = check_point.i
   extraTime = check_point.extraTime
   print ('extraTime from checkpoint = ', extraTime)
   if extraTime == nil then extraTime = 0 end
   last_curriculum_advance = check_point.last_curriculum_advance
   curriculum_level = check_point.curriculum_level
   cur_curriculum_error_rate = check_point.cur_curriculum_error_rate
   total_stdout = check_point.total_stdout
   if total_stdout == nil then total_stdout = '' end
else
   print ('CHECK POINT DOES NOT EXIST :( :( :( :( :(, length_testing = ', length_testing)
   if length_testing == 1 then
      print ('there\'s no point in doing length testing if the check point does not exist.  Goobye.')
      print ("YO?")
      -- os.exit()
   end
   -- assert (length_testing == 0) --- better be doing no length teting here.
   print ("WHAT'S GOING ON?!")
   ---- if we don't load from a check point, 
   x:copy(torch.randn(x:size())):mul(params.init_scale)
   check_point.x = x:clone():zero()
   startStep = 1
   extraTime = 0
   last_curriculum_advance = 1
   curriculum_level = 1
   cur_curriculum_error_rate = 1
   total_stdout = ''
end

if task_visualize == 1 then
   curriculum_level = #curriculum + 1000
end


local msgs_buffer = {}
local timer = torch.Timer()

print ('num_iter = ', num_iter)
print ('startStep = ', startStep)
for i = startStep, num_iter do
   if params.save_freq > 0 and i % params.save_freq == 0 and params.name ~= 'NULL' and length_testing == 0 then
      print (string.format('saving to %s', fname))
      check_point.x:copy(x)
      check_point.i = i
      check_point.last_curriculum_advance = last_curriculum_advance
      check_point.curriculum_level = curriculum_level
      check_point.cur_curriculum_error_rate = cur_curriculum_error_rate
      check_point.extraTime = timer:time().real + extraTime  -- note: the RHS's extraTime is a different var from the LHS extra 
      check_point.total_stdout = total_stdout
      assert (length_testing == 0) --- DO NOT SAVE ANYTHING if we're engaged in length_testing.  
      file.SetContents(fname, torch.serialize(check_point))
      local check_point_txt = {}
      
      local msg_txt = string.format('i=%s   curriculum_level=%s   cur_curriculum_error_rate=%s \n', i, curriculum_level, cur_curriculum_error_rate)
      assert (length_testing == 0) 
      file.SetContents(fname_txt, msg_txt)
      assert (length_testing == 0)       
      file.SetContents(fname_stdout, total_stdout)
      print (string.format('saved (including %s)!', fname_txt))
   end

    input_mem_params[{{}, 1, {}}]:fill(1) 
    efficient_mem.prepare(input_mem_params) --- 

    local difficulties = {}
    local relevant_cases = torch.zeros(batch_size)
    local cur_curriculum_levels = torch.zeros(batch_size)
    local refractory_gap = i - last_curriculum_advance

    for b = 1, batch_size do
        local cur_curriculum_pos
	local extra = 0 
	---- was 2
	while torch.random(2) == 1 do extra = extra + 1 end
	local cur_curriculum_level = curriculum_level + extra

	if cur_curriculum_level > #curriculum then --- exponential distribution.
	   cur_curriculum_level = #curriculum 
	end

        if torch.random(10) == 1 and b ~= 1 then
            cur_curriculum_pos = torch.random(#curriculum)
        elseif torch.random(4) == 1 and b ~= 1 then             
            cur_curriculum_pos = torch.random(cur_curriculum_level)
        else
            cur_curriculum_pos = cur_curriculum_level
            relevant_cases[b] = 1
        end
	if cur_curriculum_pos > #curriculum then
	   cur_curriculum_pos = #curriculum
	end
	cur_curriculum_levels[b] = cur_curriculum_pos
	assert(cur_curriculum_pos <= #curriculum)
        difficulties[b] = curriculum[cur_curriculum_pos]
        assert (difficulties[b] ~= nil)
    end


    local data_args = {seq_len, batch_size, num_symbols, difficulties}
    local input_tape, output_tape, output_mask = tasks[params.task_name](unpack(data_args))

    local inputs = {r, input_tape, input_mem_2, output_tape, output_mask, init_H, x, dx, CHECK_GRAD}
    local cur_loss, cur_zero_one_losses, cur_grads = rlntm.grad_and_loss(unpack(inputs))
    local errors = cur_grads.numbers_of_errors
    local sum_of_errors = errors[1]
    for tt = 2, #errors do
       sum_of_errors = torch.add(sum_of_errors, errors[tt])
    end
    assert (sum_of_errors:eq(cur_zero_one_losses):all())

    assert (cur_zero_one_losses:nElement() == batch_size)
    cur_zero_one_losses = cur_zero_one_losses:view(batch_size) 
    local num_preds_per_case = torch.sum(output_mask, 2)
    assert (num_preds_per_case:nElement() == batch_size)
    assert (relevant_cases:nElement() == batch_size) 
    local num_relevant_cases = torch.dot(num_preds_per_case:view(batch_size), relevant_cases)
    local num_possible_cases = num_preds_per_case:sum()
    assert (num_relevant_cases > 0)
    local cur_zero_one_loss = torch.dot(cur_zero_one_losses, relevant_cases)  / num_relevant_cases


    assert (cur_zero_one_loss <= num_relevant_cases)


    for b = 1, batch_size do
        assert (cur_zero_one_losses[b] <= torch.sum(output_mask[b]))
    end    
    assert (cur_grads.baselines ~= nil)

    ---- basically, the more evidence (bigger average), the bigger the decay. 
    local decay = params.curriculum_decay --1 - ((1 - params.curriculum_decay) * (num_relevant_cases / num_possible_cases))
    cur_curriculum_error_rate = cur_zero_one_loss * (1 - decay) + cur_curriculum_error_rate * decay

    if cur_curriculum_error_rate < params.curriculum_error_rate_thresh 
       and curriculum_level < #curriculum 
       and refractory_gap > curriculum_refractory_period --- can't upgrade the level too soon! 
    then
       local msg = 'curriculum:  success!'
       print (msg)
       total_stdout = total_stdout .. msg
       cur_curriculum_error_rate = 1
       curriculum_level = curriculum_level + 1
       last_curriculum_advance = i
       --- otherwise we're doen and there's nothing ot talk about.  
    end
    if curriculum_level > #curriculum then
        curriculum_level = #curriculum
    end

    acc_loss = acc_loss + cur_loss
    acc_batch_size = acc_batch_size + batch_size
    cur_grads.dx:mul(1/batch_size)
    for d = 1, #init_H do
        cur_grads.init_H[d]:mul(1/batch_size)
    end

    
    local norm_rl = torch.cmul(cur_grads.dx, x_rl):norm()
    if norm_rl > params.max_grad_norm_rl then
        local s = params.max_grad_norm_rl / norm_rl
        cur_grads.dx:cmul(torch.add(x_baseline, x_rl:clone():mul(s)))
    end
    local norm_baseline = torch.cmul(cur_grads.dx, x_baseline):norm()
    if norm_baseline > params.max_grad_norm_baseline then
        local s = params.max_grad_norm_baseline / norm_baseline
        cur_grads.dx:cmul(torch.add(x_rl, x_baseline:clone():mul(s)))
    end


    v:mul(momentum)
    v:add(cur_grads.dx)
    x:add(LR, v)


    if i % params.entropy_gain_factor_iter == 0 then
       ----- once we hit the entropy gain factor iter,
       ----- we drastically shrink our LR.
       entropy_gain_factor = entropy_gain_factor / 10 
       set_gain_factor(entropy_gain_factor)
       local msg = string.format('new_gain_factor = %s', entropy_gain_factor)
       print (msg)
       total_stdout = total_stdout .. (msg .. '\n')
    end

    if i % params.print_freq == 0 then
       local time_so_far = timer:time().real + extraTime

       local cases_so_far = i * batch_size
       local cases_per_sec = cases_so_far / time_so_far
       local msg = string.format('%6d:  loss = %+10.5f, norm_rl = %+10.5f, norm_baseline = %+10.5f, LR= %f, |v| = %+10.5f, ent=%+6.4f, lvl=%3d, zol=%5.4f, thresh=%5.4f, cases_per_sec = %6.4f',

				 i, acc_loss/acc_batch_size, 
				 norm_rl, norm_baseline, 
				 LR, v:norm(), 
				 entropy_gain_factor, 
				 curriculum_level, 
				 cur_curriculum_error_rate,
				 params.curriculum_error_rate_thresh,
				 cases_per_sec
       ) 

       print(msg)
       total_stdout = total_stdout .. (msg .. '\n')
       acc_loss, acc_batch_size = 0, 0
    end

    if i % vis_freq == 0 or length_testing == 1 then
       assert (batch_size >= 2)
       local ind = 2
       local msgs = visualize.printTrace(inputs, ind, mem_tape_len)
       assert (cur_curriculum_levels[ind] ~= nil)
       local msg2 = 'cur_curriculum_levels[' .. ind .. '] = ' .. cur_curriculum_levels[ind]
       table.insert(msgs, msg2)
       local msg = stringx.join('\n', msgs)
       print (msg)
       total_stdout = total_stdout .. (msg .. '\n')
    end

    if length_testing == 1 and i - startStep >= 5 then
       break
    end
end
