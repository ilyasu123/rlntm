
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

package.path = package.path .. ';rlntm_pkgs/?.lua'
--SPEED_TEST = false --- that's conclusive.  It's just slow because of the very many layers. 


local lstm = require 'lstm'
local misc = require 'misc'
require 'strict'
require 'input'
require 'output' 
require 'index'
require 'nn'
require 'nngraph'
require 'ApplyTable'
local RL = require 'RL'
require 'DoNotBackprop'
local rlntm = {}
require 'utils_2'

rlntm.num_outputs_per_step = 10
local named_zeros = misc.named_zeros

local LogSoftMax = nn.ExactLogSoftMax 

function rlntm.create(T, 
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
                      entropy_gain, 
                      separate_baseline,
                      special_init,
		      mem_write_range,
		      mem_read_range,
		      input_read_range,
		      direct_mem_input,
		      direct_input_input,
		      force_input_forward,
		      super_direct,
		      mem_size,
		      length_testing
		     )

   local gain_factor_list = {}

   if length_testing == nil then
      length_testing = 0
   end
   assert (length_testing == 0 or length_testing == 1)

   if super_direct == nil then super_direct = 0 end
   assert (super_direct == 0 or super_direct == 1)
   print ('RLNTM: super_direct = ', super_direct)

   if mem_size == nil then
      mem_size = rnn_size
   end

   if super_direct == 1 then 
      assert (num_out_symbols == emb_size)
      if mem_size == nil then mem_size = emb_size end
      assert (mem_size == emb_size)
   end

   if force_input_forward == nil then force_input_forward = 0 end
   assert (force_input_forward == 0 or force_input_forward == 1)
   print ('RLNTM: force_input_forward = ', force_input_forward)

   if direct_input_input == nil then direct_input_input = 0 end
   assert (direct_input_input == 1 or direct_input_input == 0)

   if direct_mem_input == nil then direct_mem_input = 0 end
   assert (direct_mem_input == 1 or direct_mem_input == 0)

   if mem_write_range == nil then mem_write_range = 2 end
   assert (mem_write_range >= 0)

   if mem_read_range == nil then mem_read_range = 3 end
   assert (mem_read_range >= 0)

   if input_read_range == nil then input_read_range = 2 end
   assert (input_read_range >= 0)
   
   if special_init == nil then special_init = false end
   assert (special_init == false or special_init == true)

   ---- AT PRESENT, if ENTROPY_GAIN ~= 0 then CHECK GRAD WILL FAIL.
   if entropy_gain == nil then entropy_gain = 0 end
   if separate_baseline == nil then separate_baseline = 1 end
   assert (separate_baseline == 1 or separate_baseline == 0)
   separate_baseline = (separate_baseline == 1)

   if per_step_gain == nil then per_step_gain = 0 end
   if action_slowdown == nil then action_slowdown = 1 end
   if baseline_slowdown == nil then baseline_slowdown = 1 end
   print ('special_init = ', special_init)
   print ('baseline_slowdown = ', baseline_slowdown)
   print ('entropy gain = ', entropy_gain)
   print ('separate_baseline = ', separate_baseline)
   local input_tape  = nn.Identity()()       
   local init_mem = nn.Identity()()
   local init_input_pos = nn.Identity()()
   local init_mem_pos = nn.Identity()()
   local init_H   = nn.Identity()()        
   local total_gain = nil
   local total_number_of_errors = nil

   ---- for the output tape, we need this: 
   local output_tape = nn.Identity()()
   local output_mask = nn.Identity()()
   local init_output_pos = nn.Identity()()

   local input_pos = init_input_pos
   local output_pos = init_output_pos
   local mem_pos = init_mem_pos
   local mem = init_mem
   local H = init_H
   local H_separate_baseline = init_H
   local h_separate_baseline = nil

   ---- external inputs for moving the memory
   local outputs = {}

   ---- we need a last move for every one of these guys.
   local gain, h
   local zeros = nn.ConstMul(0)(mem_pos)
   local ones = nn.ConstAdd(1)(zeros)  --- create batch_size*1 vectors of the appropriate size, full of 1s.
   local last_in_move = ones
   local last_mem_move = ones
   local last_out_move = ones
   local last_targ = ones


   local init_baseline = nil
   local init_rl = nil

   if special_init then
       init_baseline = {val = 0}
       init_rl = {val = 1}
   end

   --- step 0:  do global processing for baseline purposes.
   local H_for_init = init_H
   local input_pos_for_init = init_input_pos
   local h_out_baseline
   local forget_gate_bias_baseline = 1


   ---- start by going over the input, but in reverse, since all the action takes place
   ---- in the beginning of the tape.
   for t = max_in_len, 1, -1 do
      local input_t = nn.GetInput(){input_tape, nn.ConstAdd(t)(zeros)}   
      h_out_baseline, H_for_init = lstm.deep(depth, H_for_init, input_t, emb_size, rnn_size, forget_gate_bias_baseline, 'extra_sig_for_init', init_baseline)
   end 

   for t = 1, T do
       local input_inputs = {}
       for t = -input_read_range, input_read_range do
	  table.insert(input_inputs, nn.GetInput(){input_tape, nn.ConstAdd(t)(input_pos)})
       end

       local mem_inputs = {} 
       for t = -mem_read_range, mem_read_range do
	  ---- it's read mem and not get input beacuse we want to backprop thorugh ReadMem.
	  table.insert(mem_inputs, nn.ReadMem(){mem, nn.ConstAdd(t)(mem_pos)})
       end

       local last_moves = {
	  misc.make_emb('last_in_move',  #in_moves_table, emb_size)(last_in_move), 
	  misc.make_emb('last_mem_move', #mem_moves_table, emb_size)(last_mem_move),
	  misc.make_emb('last_out_move', #out_moves_table, emb_size)(last_out_move)
       }

       local the_input  = nn.JoinTable(2)(misc.concat(last_moves, mem_inputs, input_inputs))
       local input_width = #mem_inputs * mem_size + #last_moves*emb_size + #input_inputs*emb_size

       h, H = lstm.deep(depth, H, the_input, input_width, rnn_size, forget_gate_bias, '', init_rl)

       if separate_baseline then
            h_separate_baseline, H_separate_baseline = 
                 lstm.deep(depth, H_separate_baseline, 
			   nn.DoNotBackprop()(the_input), 
			   input_width, 
			   rnn_size, 
			   forget_gate_bias, 
			   'separate_baseline_sig', 
			   init_baseline)
       end

       ----  
       local h_for_action = h
       local h_size = rnn_size


       if direct_mem_input == 1 then
	  h_for_action = nn.JoinTable(2)(misc.concat(mem_inputs, {h_for_action}))
	  h_size = mem_size * #mem_inputs + h_size
       end

       if direct_input_input == 1 then
	  h_for_action = nn.JoinTable(2)(misc.concat(input_inputs, {h_for_action}))
	  h_size = emb_size * #input_inputs + h_size
       end



       ---- compute the (independent) distributions over the various actions that we take: 
       local length_testing_factor
       if length_testing == 0 then 
	  length_testing_factor = 1
       else 
	  length_testing_factor = 1e3
       end

       local function slow(x) return nn.ConstMul(action_slowdown * 
						  length_testing_factor)(x) end


       local in_dist_logits = slow(misc.make_lin('rlntm_in_move' , h_size, #in_moves_table , init_rl)(h_for_action))

       if force_input_forward == 0 then
	  --- if no force input forward, just do the usual thing
	  --- do nothing. 
       else
	  assert (force_input_forward == 1)
	  ---- get the constraints and apply them to the inputs.
	  local in_constraints = nn.InputActionConstraints(in_moves_table){input_tape, input_pos}
	  in_dist_logits  = nn.CAddTable(){in_dist_logits,  in_constraints}
       end
       local in_dist = LogSoftMax()(in_dist_logits)

       local mem_logits = slow(misc.make_lin('rlntm_mem_move', h_size, #mem_moves_table, init_rl)(h_for_action))
       local mem_dist = LogSoftMax()(mem_logits)

       --- add the output constraints to the output dist to rule out various actions.
       local out_constraints = nn.OutputActionConstraints(out_moves_table, T, t){output_mask, output_pos} 
       local out_logits = slow(misc.make_lin('rlntm_out_move', h_size, #out_moves_table, init_rl)(h_for_action))
       local out_dist = LogSoftMax()(nn.CAddTable(){out_logits,  out_constraints})

       local h_baseline
       if separate_baseline == true then 
           h_baseline = h_separate_baseline
       else
           h_baseline = nn.DoNotBackprop()(h)
       end
       local baseline_rep = nn.Rectifier()(nn.CAddTable(){misc.make_lin('rlntm_baseline_1', rnn_size, rnn_size, init_baseline)(h_baseline),
                                                           misc.make_lin('rlntm_baseline_2', rnn_size, rnn_size, init_baseline)(h_out_baseline)})
                                      
       local baseline_t = nn.ConstMul(baseline_slowdown)(misc.make_lin('rlntm_baseline_3', rnn_size, 1, init_baseline)(baseline_rep))

       ----- sample the move: 
       ----- NOTE:  it is acceptable to use the same baseline for all actions since they all get an
       -----        identical future reward.
       local in_move  = RL.Sample(in_moves_table ){in_dist,  baseline_t}
       local mem_move = RL.Sample(mem_moves_table){mem_dist, baseline_t}
       local out_move = RL.Sample(out_moves_table){out_dist, baseline_t}

       ---- collect all the moves in order to emit them at the end.
       table.insert(outputs, in_move) 
       table.insert(outputs, mem_move)
       table.insert(outputs, out_move)

       table.insert(outputs, input_pos) 
       table.insert(outputs, mem_pos)
       table.insert(outputs, output_pos)

       in_move = nn.DoNotBackprop()(in_move)
       mem_move = nn.DoNotBackprop()(mem_move)
       out_move = nn.DoNotBackprop()(out_move)

       local function ent(dist)
           local const_mul = nn.ConstMul(-entropy_gain)
	   table.insert(gain_factor_list, const_mul)
	   return const_mul(nn.Sum(2)(nn.CMulTable(){nn.Exp()(dist), dist}))
       end

       ---- write to adjacent memory addresses:
       for t = -mem_write_range, mem_write_range do
	  local f = nn.Sigmoid()(
	     nn.ConstAdd(forget_gate_bias)(
		misc.make_lin('rlntm_f_'..t, h_size, mem_size, init_rl)(h_for_action)))

	  local w = nn.Tanh()(   
	     misc.make_lin('rlntm_w_'..t, h_size, mem_size, init_rl)(h_for_action))

	  mem = nn.WriteMem(){mem, nn.ConstAdd(t)(mem_pos), f, w} 

	  if super_direct == 1 then --- add some of the inputs in the current view to the memory
	     for t2 = -input_read_range, input_read_range do
		local p = nn.Sigmoid()(nn.ConstAdd(forget_gate_bias)(misc.make_lin('rlntm_m_p_'..t..'_'..t2, h_size, 1, init_rl)(h_for_action)))
		local ii = nn.GetInput(){input_tape, nn.ConstAdd(t2)(input_pos)}

		local iip = nn.EltwiseVecTimesMul(){p, ii}

		local p2 = nn.Sigmoid()(nn.ConstAdd(forget_gate_bias)(misc.make_lin('rlntm_m_p2_'..t..'_'..t2, h_size, 1, init_rl)(h_for_action)))

		mem = nn.WriteMem(){mem, nn.ConstAdd(t)(mem_pos), nn.ExpandVec(mem_size)(p2), iip}
	     end
	  end
       end



       local logits_t = misc.make_lin('rlntm_pred', h_size, num_out_symbols, init_rl)(h_for_action)
       ----add a bunch of things directly to the logistic
       if super_direct == 1 then
	  for t = -mem_write_range, mem_write_range do
	     local p = nn.Sigmoid()(nn.ConstAdd(forget_gate_bias)(misc.make_lin('rlntm_m_out_'..t, h_size, 1, init_rl)(h_for_action)))
	     local ii = nn.ReadMem(){mem, nn.ConstAdd(t)(mem_pos)}
	     local iip = nn.EltwiseVecTimesMul(){p, ii}
	     logits_t = nn.CAddTable(){logits_t, iip}
	  end
	  for t = -input_read_range, input_read_range do
	     local p = nn.Sigmoid()(nn.ConstAdd(forget_gate_bias)(misc.make_lin('rlntm_in_out_'..t, h_size, 1, init_rl)(h_for_action)))
	     local ii = nn.GetInput(){input_tape, nn.ConstAdd(t)(input_pos)}
	     local iip = nn.EltwiseVecTimesMul(){p, ii}
	     logits_t = nn.CAddTable(){logits_t, iip}
	  end
       end


       local pred_t = LogSoftMax()(logits_t)
       local targ_t = nn.GetInput(){output_tape, output_pos}
       local mask_t = nn.GetInput(){output_mask, output_pos}
       local true_mask_t = nn.CMulTable(){nn.DoNotBackprop()(out_move), mask_t}
       local remaining_gains_t = nn.ConstMul(per_step_gain)(nn.RemainingOutputs(){output_mask, output_pos})
       local unmasked_gain_t = nn.MyIndex(){pred_t, targ_t}
       local entropy_gain_t  = nn.CAddTable(){ent(in_dist), ent(mem_dist), ent(out_dist)}
       local masked_gain_t = nn.CMulTable(){unmasked_gain_t, true_mask_t}
       ----local gain_t   = nn.CAddTable(){entropy_gain_t, masked_gain_t, remaining_gains_t} ---- UNCOMMENT TO PASS CHECK GRAD
       ----and yet, check grad passes even with this.  
       local gain_t   = nn.CAddTable(){masked_gain_t, remaining_gains_t}

       --- NOTE the elegant and slightly nonobvious fact: out_move = 0 or 1. We predict = experience a gain when out_move = 1.

       ---- apply the move after making the prediction, as it plays better with the visualizer.
       input_pos = nn.MovePtr(){input_pos, in_move}
       mem_pos = nn.MovePtr(){mem_pos, mem_move}
       output_pos = nn.MovePtr(){output_pos, out_move}

       ---- accumulate the total loss.   
       if total_gain == nil 
       then total_gain = nn.Identity()(gain_t)
       --else total_gain = nn.CAddTable(){total_gain, gain_t} 
       else total_gain = nn.CAddTable(){total_gain, gain_t, entropy_gain_t} --- entropies go here.
       end

       local number_of_errors = nn.ZeroOneLoss(){pred_t, targ_t, true_mask_t}
       if total_number_of_errors == nil 
       then total_number_of_errors = nn.Identity()(number_of_errors)
       else total_number_of_errors = nn.CAddTable(){total_number_of_errors, number_of_errors}
       end

       table.insert(outputs, pred_t)
       table.insert(outputs, baseline_t)

       ---- gain_t must be the last one to be inserted to this table! 
       table.insert(outputs, number_of_errors)
       table.insert(outputs, gain_t)


       --- keep track of the previous actions, because the previous action is highly useful
       last_in_move = nn.ApplyTable(misc.inv_table(in_moves_table))(in_move)
       last_mem_move = nn.ApplyTable(misc.inv_table(mem_moves_table))(mem_move)
       last_out_move = nn.ApplyTable(misc.inv_table(out_moves_table))(out_move)
       last_targ = targ_t
   end 

   local inputs = misc.concat({input_tape, init_mem, output_tape, output_mask, init_input_pos, init_mem_pos, init_output_pos, init_H}) 

   --- the sum must be at the end since it returns a scalar.  
   local function set_gain_factor(gain)
      for i,f in ipairs(gain_factor_list) do
	 gain_factor_list.multiplier = gain
      end
   end

   local final_outputs = misc.concat({total_gain, total_number_of_errors}, outputs)
   return nn.gModule(inputs, final_outputs), set_gain_factor
end

local S = 'grad_and_loss_'
function rlntm.grad_and_loss(r, input_tape, input_mem, output_tape, output_mask, init_H, x, dx, CHECK_GRAD)
    --- though we return negloss, i.e., gain.
    
    dx:zero()
    local batch_size = input_tape:size(1) 
    local init_input_pos = named_zeros(S..'init_input_pos', batch_size):fill(1)
    local init_mem_pos = named_zeros(S..'init_mem_pos', batch_size):fill(1)    
    local init_output_pos = named_zeros(S..'init_output_pos', batch_size):fill(1)

    local inputs = {input_tape, input_mem, output_tape, output_mask, init_input_pos, init_mem_pos, init_output_pos, init_H}
    local outputs = r:forward(inputs)

    local losses = outputs[1]
    local zero_one_losses = outputs[2] 

    assert ((#outputs - 2) % rlntm.num_outputs_per_step == 0, "terrible inconsistency")
    local T = (#outputs - 2) / rlntm.num_outputs_per_step --- outputs = (losses, a1, b1, c1, l1,  a2, b2, c2, l2, ....)

    ---- compute cumulative future losses needed for reinforce: 
    local cuml_future_reward = {}
    for t = T, 1, -1 do
        local reward_pos = 2 + t * rlntm.num_outputs_per_step 
        ---- assumes that gain_t is at the end of each batch of outputs.
	cuml_future_reward[t] = named_zeros(S..'cuml_future_reward_'..t, 
					    outputs[reward_pos]:size())

        if t == T 
        then cuml_future_reward[t]:copy(outputs[reward_pos])
        else cuml_future_reward[t]:add(cuml_future_reward[t+1], outputs[reward_pos])
        end
    end

    local gradOutput = {}
    local gradOutput_coverage = {}
    if CHECK_GRAD ~= true then
       gradOutput[1] = named_zeros(S .. 'gradOutput[1]', outputs[1]:size()):fill(1)
       gradOutput_coverage[1] = true
    else
        assert (outputs[1]:nDimension() == 1)
        assert (outputs[1]:size(1) == global_action_probs:size(1))
	gradOutput[1] = named_zeros(S .. 'gradOutput[1]', global_action_probs:size()):fill(1)
        gradOutput[1]:copy(global_action_probs)
	gradOutput_coverage[1] = true
    end
    if mode == 'hack' then
       gradOutput[2] = "nil"
       gradOutput_coverage[2] = true
    else
       gradOutput[2] = named_zeros(S .. 'gradOutput[2]', outputs[2]:size())
       gradOutput_coverage[2] = true
    end 
    assert (gradOutput[2] ~= nil)

    local pos = 0
    local baselines = {}
    local gains = {}
    local numbers_of_errors = {}
    for t = 3, #outputs, rlntm.num_outputs_per_step do
        pos = pos + 1
        local r_future = cuml_future_reward[pos]   
	local c = 0
	local set_of_ts = {}

        for tt = t, t+2 do
            assert (r_future ~= nil)
            gradOutput[tt] = r_future
	    gradOutput_coverage[tt] = true
	    assert (gradOutput[tt] ~= nil)
	    c = c + 1
	    set_of_ts[tt] = true
        end
	local num = 0
	for u,v in pairs(set_of_ts) do num = num + 1 end

	if c ~= 3 then
	   print ('WTF')
	   print ('c should be 3')
	   print ('yet c = ', c)
	   print ('also, t = ', t)
	   print ('oh, and num = ', num)
	   print ('and set_of_ts = ', set_of_ts)
	   print ('so odd.')
	   print ('this is a bad nondeterministic bug.')
	end
	assert (c == 3)
        for tt = t + 3, t + rlntm.num_outputs_per_step - 1 do
	   c = c + 1
           if mode == 'hack' then
               gradOutput[tt] = "nil"
           else
               gradOutput[tt] = named_zeros(S .. 'gradOutput_'.. tt, outputs[tt]:size())
	       assert (gradOutput[tt] ~= nil)
           end
	   gradOutput_coverage[tt] = true
           assert (gradOutput[tt] ~= nil)
        end
	assert (c == rlntm.num_outputs_per_step)
	for tt = t, t + rlntm.num_outputs_per_step - 1 do
	   assert (gradOutput_coverage[tt] == true)
	   assert (gradOutput[tt] ~= nil)
	end
	local output_t_7 = named_zeros(S .. 'outputs_' .. t+7, outputs[t+7]:size())
        table.insert(baselines, output_t_7:copy(outputs[t + 7]))

	local output_t_num_outputs_m2 = named_zeros(S .. 'outputs_' .. t + rlntm.num_outputs_per_step - 2, 
						    outputs[t + rlntm.num_outputs_per_step - 2]:size())
	local output_t_num_outputs_m1 = named_zeros(S .. 'outputs_' .. t + rlntm.num_outputs_per_step - 1, 
						    outputs[t + rlntm.num_outputs_per_step - 1]:size())

        table.insert(numbers_of_errors, output_t_num_outputs_m2:copy(outputs[t + rlntm.num_outputs_per_step - 2]))
        table.insert(gains,             output_t_num_outputs_m1:copy(outputs[t + rlntm.num_outputs_per_step - 1]))
        assert(2 + pos * rlntm.num_outputs_per_step == t + rlntm.num_outputs_per_step - 1)  
    end 

    for tt = 1, #gradOutput do
        if (gradOutput[tt] == nil) then
	   print ('gradOutput_coverage[', tt, '] = ', gradOutput_coverage[tt])
	   print ('tt = ', tt, 'results in gradOutput[tt] being nil.  #gradOutput = ', #gradOutput)
        end
        assert (gradOutput[tt] ~= nil)
    end

    local floss = {}
    floss[T] = named_zeros(S .. 'floss_' .. T, gains[T]:size())
    floss[T]:copy(gains[T])
    local t = T 
    for t = T-1, 1, -1 do
       floss[t] = named_zeros(S .. 'floss_' .. t, gains[t]:size())
       floss[t]:add(gains[t], floss[t+1])
    end                 
    if (#gradOutput ~= #outputs) then
        print ('#gradOutput = ', #gradOutput)
        print ('#outputs = ', #outputs) 
    end
    assert (#gradOutput == #outputs)
    

    local gradInputs = r:backward(inputs, gradOutput)
    local grad_input_tape, grad_input_mem, grad_output_tape, grad_output_mask, grad_init_input_pos, grad_init_mem_pos, grad_init_output_pos, grad_init_H  = unpack(gradInputs)
    local relevant_grad_inputs = {grad_input_mem, grad_init_H}

    local total_loss 
    local total_zero_one_loss
    if CHECK_GRAD ~= true then
       total_loss = losses:sum()
       total_zero_one_loss = zero_one_losses --:sum()
    else
       total_loss = torch.dot(losses, global_action_probs)
       total_zero_one_loss = zero_one_losses --torch.dot(zero_one_losses, global_action_probs)
    end
    return total_loss,  total_zero_one_loss,  
           {dx = dx, input_mem = grad_input_mem, init_H = grad_init_H, baselines = baselines, 
           gains = gains,
	   numbers_of_errors = numbers_of_errors,
           floss = floss}, outputs
end

assert (rlntm.num_outputs_per_step ~= nil)
return rlntm

