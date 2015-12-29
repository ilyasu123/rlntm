
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

local rlntm = require 'rlntm_pkgs.rlntm'
local stringx = require 'pl.stringx'
local visualize = {}

visualize.char_map = {[26] = '_', [25] = ' ', [1] = '1', [2] = '2', [3] = '3', [4] = '4', [5] = '5', [6] = '6', [7] = '7', [8] = '8', [9] = '9', [10] = '0', [11] = '(', [12] = ')', [13] = '+', [14] = '-', 
   [15] = '*', [16] = '/', [17] = 'a', [18] = 'b', [19] = 'c', [20] = 'd', [21] = 'e', [22] = 'f', [23] = 'h', [24] = 'g', [27] = '-', [28] = '.', [29] = '=', [30] = '?', [31] = 'A', [32] = 'B', [33] = 'C', [34] = 'D', [35] = 'E'}
visualize.inv_char_map = {}
for a, b in pairs(visualize.char_map) do
    visualize.inv_char_map[b] = a
end


function visualize.printTrace(inputs, batch_id, mem_len)
    if batch_id == nil then batch_id = 1 end
    local r, input_tape, input_mem, output_tape, output_mask, init_H, x, dx, CHECK = unpack(inputs)
    if (input_tape:size(3) > #visualize.char_map) then
        print ('input_tape:size = ', unpack(input_tape:size():totable()))
        print ('#visualize.char_map = ', #visualize.char_map)
    end
    assert (input_tape:size(3) <= #visualize.char_map) ---the third dimension is the width of the input.
    local loss, zero_one_loss, cur_grads, rlntm_outputs = rlntm.grad_and_loss(unpack(inputs))
    -- format of outputs:
    -- total loss, a1, b1, c1, input_pos_1, mem_pos_1, output_pos_1, pred_1, loss_1, a2, b2, c2, input_pos_2, mem_pos_2, output_pos_2, pred2, loss2, a3, b3, c3, ...
    -- OK.  Let's print it.
    local TT = #rlntm_outputs
    assert (rlntm.num_outputs_per_step ~= nil)
    local outputs_per_step = rlntm.num_outputs_per_step 
    assert (outputs_per_step ~= nil)
    assert ((TT - 2) % outputs_per_step == 0)
    local T = (TT - 2) / outputs_per_step
    local input_len = input_tape:size(2)
    local output_len = output_mask:sum(2)[{batch_id,1}]
    --local mem_len  -- it's already given to us

    local input_arr = torch.zeros(T + 1, input_len):fill(visualize.inv_char_map[' '])
    local output_arr = torch.zeros(T + 1, output_len):fill(visualize.inv_char_map[' '])
    local mem_arr = torch.zeros(T + 1, mem_len):fill(visualize.inv_char_map[' '])

    ---- first, populate the input:
    for i = 1, input_len do
        local inp = ({input_tape[{batch_id, i}]:max(1)})[2][1]
        input_arr[{1, i}] = inp
    end 
    for t = 1, T do
        local pos = rlntm_outputs[2 + outputs_per_step*(t-1) + 4][batch_id]
        pos = (pos-1) % input_len + 1
        local inp = input_arr[{1, pos}]
        input_arr[{t + 1, pos}] = inp
    end

    ---- next, populate the output: 
    for i = 1, output_len do
        output_arr[{1, i}] = output_tape[{batch_id, i}]
    end
    for t = 1, T do 
        local pos = rlntm_outputs[2 + outputs_per_step*(t-1) + 6][batch_id]
        pos = (pos-1) % output_len + 1
        local preds = rlntm_outputs[2 + outputs_per_step*(t-1) + 7][batch_id]  --- batch_size * preds (preds = LogSoftMax)
        local move = rlntm_outputs[2 + outputs_per_step*(t-1) + 3][batch_id]  --- batch_size  -- the move
        assert (preds:nDimension() == 1)
        local pred = ({preds:max(1)})[2][1]
        if move == 1 then
            output_arr[{t + 1, pos}] = pred
        else
            output_arr[{t + 1, pos}] = visualize.inv_char_map['_']
        end
    end

    ---- next, populate the memory:
    for t = 1, T do
        local pos = rlntm_outputs[2 + outputs_per_step*(t-1) + 5][batch_id]
        pos = (pos-1) % mem_len + 1
        mem_arr[{t + 1, pos}] = 15
    end


    ---- finally, lets print this:
    local baselines = cur_grads.baselines
    local floss = cur_grads.floss
    local gains = cur_grads.gains
    local errors = cur_grads.numbers_of_errors

    local msgs = {}
    local char_map = visualize.char_map
    for t = 1, T + 1 do
        local line = {}
        for i = 1, input_len do
            table.insert(line, char_map[input_arr[{t, i}]])
        end         
        for i = 1, 4 do
            table.insert(line, ' ') if i == 2 then table.insert(line, '|') end
        end
        for i = 1, output_len do
            table.insert(line, char_map[output_arr[{t, i}]])
        end
        for i = 1, 4 do 
            table.insert(line, ' ') if i == 2 then table.insert(line, '|') end
        end
        for i = 1, mem_len do
            table.insert(line, char_map[mem_arr[{t, i}]])
        end

        if t > 1 then
             table.insert(line, '       ')
             table.insert(line, string.format('b=%+9.5f fl=%+9.5f g=%+9.5f z=%d', baselines[t-1][{batch_id,1}], floss[t-1][batch_id], gains[t-1][batch_id], errors[t-1][{batch_id,1}]))
        end
	local cur_line = stringx.join('', line) 
	table.insert(msgs, cur_line)
    end
    return msgs
end

return visualize
