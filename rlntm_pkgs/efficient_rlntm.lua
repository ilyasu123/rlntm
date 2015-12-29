
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

package.path = package.path .. ';rlntm_pkgs/?.lua' 

CHECK_GRAD = false --- by default, no check grad 

local lstm = require 'lstm'
local misc = require 'misc'
require 'strict'
require 'input'
require 'output' 
require 'index'
require 'nn'
require 'nnd'
require 'nngraph'
require 'ApplyTable'
local RL = require 'RL'
require 'DoNotBackprop'
local rlntm = {}

rlntm.num_outputs_per_step = 10
local named_zeros = misc.named_zeros

---- TODO:  write a number of helpful GPU-Friendly layers
---- nn.InPlaceJoinTable
---- nn.InPlaceSplitTensor --- input: fixed sizes, dynamic tensor --> table of dynamic tensors of appropriate sizes
local function pieces_to_dict(pieces)
   local ans = {}
   ans.actions = {}
   local ptr = 0

   for j = 1, 3 do -- 1,3 = the three different actions
      ans.actions[ptr + j] = pieces[ptr + j]
   end
   ptr = ptr + 3


   ans.output = pieces[ptr + 1] ---- the output softmax
   ptr = ptr + 1

   assert ((#pieces - 4) % 2 == 0) 
   local mem_range = (#pieces - 4) / 2 - 1 ---- 

   ans.mem_forget = {}
   for j = 1, (2*mem_range+1) do
      ans.mem_forget[ptr + j] = pieces[ptr + i]
   end
   ptr = ptr + (2*mem_range + 1)

   ans.mem_write = {}
   for j = 1, (2*mem_range+1) do
      ans.mem_write[ptr  +j] = pieces[ptr + j]
   end
   ptr = ptr + (2*mem_range + 1)

   assert (ptr == #pieces)
   return ans
end

local function dict_to_pieces(dict)
   local ans = {}
   for _,p in ipairs(dict.actions) do
      table.insert(ans, p)
   end
   table.insert(ans, dict.output)
   for _,p in ipairs(dict.mem_forget) do
      table.insert(ans, p)
   end
   for _,p in ipairs(dict.mem_write) do
      table.insert(ans, p)
   end
   return ans
end

function rlntm.create_controller(rnn_size, 
				 big_input_size,
				 depth,
				 num_symbols,
				 forget_gate_bias,
				 mem_range,
				 action_slowdown)

   local big_input = nn.Identity()()
   local mem_input = nn.Identity()()
   local prev_H = nn.Identity()()
   
   local the_input = nn.JoinTable(){big_input, mem_input}
   local the_input_size = big_input_size + (2*mem_range + 1)*rnn_size

   local h_out, next_H = lstm.deep(depth, prev_H, the_input, 
				   the_input_size, rnn_size, forget_gate_bias, 'controller')

   local table_of_output_sizes = {3, 3, 2, num_symbols}
   for j = 1, 2 * (2*mem_range + 1) do
      table.insert(table_of_output_sizes, rnn_size)
   end
   local output_size = misc.sum(table_of_output_sizes)



   local big_output = misc.make_lin('big_output', h_size, output_size)(h_out)

   ---- Next step:  break the big output into small pieces, apply the appropriate softmax
   ---- to each piece, and concatenate it all back
   local pieces = nn.InPlaceSplitTensor(table_of_output_sizes)(big_output)
   local pieces_dict = pieces_to_dict(pieces)
   for i = 1, #pieces_dict.actions  do
      pieces_dict.actions[i] = nnd.ExactLogSoftMax()(nnd.ConstMul(action_slowdown)(pieces_dict.actions[i]))
   end
   pieces_dict.output = nnd.ExactLogSoftMax()(pieces_dict.output) 
   for i = 1, #pieces_dict.mem_forget do
      pieces_dict.mem_forget[i] = nn.Sigmoid()(pieces_dict.mem_forget[i])
   end
   for i = 1, #pieces_dict.mem_write do
      pieces_dict.mem_write[i] = nn.Tanh()(pieces_dict.mem_write[i])
   end
   local final_output = nn.InPlaceJoinTable()(dict_to_pieces(pieces_dict))

   return nn.gModule({input, prev_H}, {final_output})
end

function rlntm.create_shell(rnn_size, num_symbols, mem_range, in_moves_table, mem_moves_table, out_moves_table)
   local table_of_output_sizes = {3, 3, 2, num_symbols}
   for j = 1, 2 * (2*mem_range + 1) do
      table.insert(table_of_output_sizes, rnn_size)
   end

   ----- shell is a CPU object.  We don't do GPU computations here.  Everything is CPU only.
   local controller_output = nn.Identity()()
   local baseline = nn.Identity()()   ------ the baselines come from a separate model.
   local dict = pieces_to_dict(nn.InPlaceSplitTensor(table_of_output_sizes)(controller_output))
   local mem = nn.Identity()()
   local mem_pos = nn.Identity()()

   ----- baselines need to come from someplace.  They'll come out of the input. That's OK. 
   local in_move  = RL.Sample(in_moves_table){dict.actions[1],  baseline}
   local mem_move  = RL.Sample(mem_moves_table){dict.actions[2],  baseline}

   local out_constraints = nn.Identity()() 
   --- the constraints over the output action are 

   local out_move  = RL.Sample(out_moves_table){dict.actions[3],  baseline}

   ---- collect all the moves in order to emit them at the end.
   local outputs = {} ---- we must collect the outputs right here, at once. 
   table.insert(outputs, in_move) 
   table.insert(outputs, mem_move)
   table.insert(outputs, out_move)

   local function ent(dist)
      return nnd.ConstMul(-entropy_gain)(nn.Sum(2)(nn.CMulTable(){nn.Exp()(dist), dist}))
   end

   ---- pred is our prediciton, targ, mask, and remaining_outputs come form the wrapping code
   local pred = dict.output 
   local targ = nn.Identity()() --nn.GetInput(){output_tape, output_pos}
   local mask = nn.Identity()() --nn.GetInput(){output_mask, output_pos}
   local remaining_outputs = nn.Identity()() --- nn.RemainingOutputs(){output_mask, output_pos}
   local true_mask = nn.CMulTable(){nn.DoNotBackprop()(out_move), mask}
   local remaining_gains = nnd.ConstMul(per_step_gain)(remaining_outputs)
   local unmasked_gain = nn.MyIndex(){pred, targ}
   local entropy_gain  = nn.CAddTable(){ent(in_dist), ent(mem_dist), ent(out_dist)}
   local masked_gain = nn.CMulTable(){unmasked_gain, true_mask}

   ----local gain_t   = nn.CAddTable(){entropy_gain_t, masked_gain_t, remaining_gains_t} ---- UNCOMMENT TO PASS CHECK GRAD
   ----and yet, check grad passes even with this.  
   local gain   = nn.CAddTable(){masked_gain, remaining_gains}

   ---- OK, good:  we write the good mem here.
   assert (#dict.forget == 2*mem_range + 1)
   assert (#dict.write == 2*mem_range + 1)
   for i = 1, 2*mem_range+1 do
      local f = dict.forget[i]
      local w = dict.write[i]
      mem = nn.WriteMem(){mem, nnd.ConstAdd(i - mem_range - 1)(mem_pos), f, w} 
   end

   table.insert(outputs, gain)
   table.insert(outputs, mem) 
   --- TODO:  update mem_pos outside of mem.  Okay.   Let's go over the logic of this thing again.

   return nn.gModule({controller_output, baseline, targ, mask, remaining_outputs}, outputs)
end

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
		      mem_range,
		      mem_read_range,
		      input_read_range)

   local big_input_size     ---- TODO: compute big input size

   local controller = rlntm.create_controller(rnn_size, 
					      big_input_size,
					      depth,
					      num_symbols,
					      input_read_range,
					      forget_gate_bias,
					      input_wrtie_range,
					      mem_range,
					      action_slowdown)
   if CHECK_GRAD == false then
      controller = controller:cuda()
   end
   local c_x, c_dx = controller:getParameters()
   local model = {}
   model.controllers = nnd.cloneManyTimes(T)(controller)
   
   local shell = rlntm.create_shell(rnn_size, num_symbols, mem_range, in_moves_table, mem_moves_table, out_moves_table)
   local s_x, s_dx = shell:getParameters()
   model.shells = nnd.cloneManyTimes(T)(shells)

   model.c_x, model.c_dx = c_x, c_dx
   model.s_x, model.s_dx = s_x, s_dx
   model.T = T

   ---- I JUST REALIZED:  WE DON'T NEED EMBEDDINGS TO REPRESENT THE VARIOUS ACTIONS!  GOOD! 
   return model
end

function rlntm.grad_and_loss(model, input_tape, input_mem, output_tape, output_mask, init_H, CHECK_GRAD)
   model.c_dx:zero()
   model.s_dx:zero()

   local batch_size = input_tape:size(1) 
   local init_input_pos = torch.zeros(batch_size):fill(1) 
   local init_mem_pos = torch.zeros(batch_size):fill(1)    
   local init_output_pos = torch.zeros(batch_size):fill(1)

   
   ---- begin with the forward pass
   local H = init_h
   local shell_outputs = {}
   local inputs_to_controllers = {}
   local inptus_to_shells = {}
   for t = 1, model.T do 
      ----- TODO here:  produce the inputs to all controllers. 
      controller_inputs = {}


      inputs_to_controllers[t] = {H, big_concat_of_inputs}
      local final_output = unpack(model.controllers[t]:forward(inputs_to_controllers[t]))
      inputs_to_shells[t] = {final_outptu:double(),  ---- move final output to host
			     baseline,               ---- get baseline from someplace
			     targ, 
			     mask,
			     remaining_outputs}

      shell_outputs[t] = model.shells[t]:forward(inputs_to_shells[t])
      ----- at this point, I need to do a bit of logicing:
      ----- 
   end

end
