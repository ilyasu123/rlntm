
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--


local tasks = {}
local visualize = require 'rlntm_pkgs.visualize'
local stringx = require 'pl.stringx'
require 'strict'

tasks.long_integer_addition_curriculum = {}
for i = 1, 8 do
    table.insert(tasks.long_integer_addition_curriculum, i)
end   
function tasks.long_integer_addition(absolute_max_len, batch_size, num_syms, difficulties)
   local function gen_one_example(b)         

      local max_len = difficulties[b]
      if max_len > absolute_max_len then
	 max_len = absolute_max_len
      end
      assert (max_len ~= nil)
      assert (max_len <= 10)
      local len_1 = torch.random(max_len)
      local len_2 = torch.random(max_len)
      local num_1 = torch.random(10^len_1)
      local num_2 = torch.random(10^len_2)
      local num_3 = num_1 + num_2
      local seq = {}
      local num_1_spell = ''..num_1
      local num_2_spell = ''..num_2
      local num_3_spell = ''..num_3

      local num_digits_1 = #num_1_spell
      local num_digits_2 = #num_2_spell
      local num_digits_3 = #num_3_spell

      for i = 1, num_digits_1 do
          table.insert(seq, string.sub(num_1_spell, i, i))
      end
      table.insert(seq, '+')
      for i = 1, num_digits_2 do
          table.insert(seq, string.sub(num_2_spell, i, i))
      end
      return seq, num_3      
   end

   local input_tape_len = 4 * absolute_max_len
   local input_tape = torch.zeros(batch_size, input_tape_len, num_syms)
   local output_tape = torch.zeros(batch_size, 16)
   local output_mask = torch.zeros(batch_size, 16)

   for b = 1, batch_size do
       local seq, ans = gen_one_example(b)
       while (#seq > input_tape_len) do
           seq, ans = gen_one_example()
       end
       local p = 0
       for pp = 1, #seq do
           p = p + 1
           local sym = visualize.inv_char_map[''..seq[p]]
           assert (sym ~= nil)
           if sym == nil then
              print ('uh oh')
              print ('seq = ', seq, 'seq[p] = ', seq[p], 'p= ', p)
           end
           assert (sym > 0)
           assert (sym <= num_syms)
           input_tape[{b, p, sym}] = 1
       end
       p = p + 1
       input_tape[{b, p, visualize.inv_char_map['=']}] = 1
       for pp = #seq + 2, input_tape_len do
           p = p + 1
           input_tape[{b, p, visualize.inv_char_map[' ']}] = 1
       end
       local ans_spell = ''..ans
       local num_digits = #ans_spell
       assert (num_digits <= output_mask:size(2))
       local pos = 1
       local start = 1
       
       for d = start, num_digits do
           output_tape[{b, pos}] = visualize.inv_char_map[string.sub(ans_spell, d, d)]
           output_mask[{b, pos}] = 1
           pos = pos + 1
       end
       output_tape[{b, pos}] = visualize.inv_char_map['.']
       output_mask[{b, pos}] = 1
       pos = pos + 1

       while pos <= output_mask:size(2) do
           output_tape[{b, pos}] = 1
           output_mask[{b, pos}] = 0
           pos = pos + 1
       end 
   end 
   return input_tape, output_tape, output_mask
end






tasks.additionBase3_curriculum = {}
for i = 1, 18 do
    table.insert(tasks.additionBase3_curriculum, i)
end   

function tasks.additionBase3(absolute_max_len, batch_size, num_syms, difficulties)
   local function toBase3Str(int)
      local revans = {}
      while math.floor(int) > 0 do 
	 table.insert(revans, '' .. int % 3)
	 int = math.floor(int / 3)
      end
      if revans == {} then table.insert(revans, '0') end
      local ans = {}
      local p = 1
      for i = #revans, 1, -1 do
	 ans[p] = revans[i]
	 p = p + 1
      end
      return stringx.join('', ans)
   end

   local function gen_one_example(b)         

      local max_len = difficulties[b]
      if max_len > absolute_max_len then
	 max_len = absolute_max_len
      end
      assert (max_len ~= nil)
      assert (max_len <= 19)
      local len_1 = torch.random(max_len)
      local len_2 = torch.random(max_len)
      local num_1 = torch.random(3^len_1)
      local num_2 = torch.random(3^len_2)
      local num_3 = num_1 + num_2
      local seq = {}
      local num_1_spell = toBase3Str(num_1)
      local num_2_spell = toBase3Str(num_2)
      local num_3_spell = toBase3Str(num_3)

      local num_digits_1 = #num_1_spell
      local num_digits_2 = #num_2_spell
      local num_digits_3 = #num_3_spell

      for i = 1, num_digits_1 do
          table.insert(seq, string.sub(num_1_spell, i, i))
      end
      table.insert(seq, '+')
      for i = 1, num_digits_2 do
          table.insert(seq, string.sub(num_2_spell, i, i))
      end
      return seq, num_3_spell
   end

   local input_tape_len = 4 * absolute_max_len
   local input_tape = torch.zeros(batch_size, input_tape_len, num_syms)
   local output_tape = torch.zeros(batch_size, 16)
   local output_mask = torch.zeros(batch_size, 16)

   for b = 1, batch_size do
       local seq, ans_spell = gen_one_example(b)
       while (#seq > input_tape_len) do
           seq, ans_spell = gen_one_example()
       end
       local p = 0
       for pp = 1, #seq do
           p = p + 1
           local sym = visualize.inv_char_map[seq[p]]
           assert (sym ~= nil)
           if sym == nil then
              print ('uh oh')
              print ('seq = ', seq, 'seq[p] = ', seq[p], 'p= ', p)
           end
           assert (sym > 0)
           assert (sym <= num_syms)
           input_tape[{b, p, sym}] = 1
       end
       p = p + 1
       input_tape[{b, p, visualize.inv_char_map['=']}] = 1
       for pp = #seq + 2, input_tape_len do
           p = p + 1
           input_tape[{b, p, visualize.inv_char_map[' ']}] = 1
       end
       local num_digits = #ans_spell
       assert (num_digits <= output_mask:size(2))
       local pos = 1
       local start = 1
       
       for d = start, num_digits do
           output_tape[{b, pos}] = visualize.inv_char_map[string.sub(ans_spell, d, d)]
           output_mask[{b, pos}] = 1
           pos = pos + 1
       end
       output_tape[{b, pos}] = visualize.inv_char_map['.']
       output_mask[{b, pos}] = 1
       pos = pos + 1

       while pos <= output_mask:size(2) do
           output_tape[{b, pos}] = 1
           output_mask[{b, pos}] = 0
           pos = pos + 1
       end 
   end 
   return input_tape, output_tape, output_mask
end





tasks.nested_modular_curriculum = {} 
for max_depth = 1, 6 do 
    for max_num = 1, 10 do
       table.insert(tasks.nested_modular_curriculum, {max_nums = max_num, max_depth = max_depth })
    end
end

function tasks.nested_modular(absolute_max_len, batch_size, num_syms, difficulties) 
   if difficulties == nil then 
       difficulties = {}
       for i = 1, batch_size do
           table.insert(difficulties, {max_nums = 2, max_depth = 1})
       end
   end

   local number, operator, open_paren, closed_paren = 1, 2, 3, 4
   local state_trans = { [number] = {operator, closed_paren}, 
                         [operator] = {number, open_paren},
                         [open_paren] = {number, open_paren},
                         [closed_paren] = {operator, closed_paren} }

   local done_state_trans = { [number] = {closed_paren}, 
                              [operator] = {number},
                              [open_paren] = {number},
                              [closed_paren] = {operator, closed_paren} }


   local function gen_one_example(b) 
      local max_nums = difficulties[b].max_nums
      local max_depth = difficulties[b].max_depth
      assert (max_nums ~= nil)
      assert (max_depth ~= nil)

      local ans = {}
      local state = ({number, open_paren})[torch.random(2)]
      if max_depth == 1 then 
          state = number
      end
      local done = false
      local len = 0
      local numerical_ans_stack = {0}
      local num_numbers = 0
      local stack_depth = 1
      local prev_op_stack = {'+'}
      local num_open_parens = 0
      assert (absolute_max_len > 5)
      local max_len = torch.random(absolute_max_len - 2) + 2
      local can_be_done = false
      local state_sequence = {}
      local stack_depth_sequence = {}
      local keep_going = true
      while not done do
          assert (stack_depth >= 1)
          table.insert(state_sequence, state)
          table.insert(stack_depth_sequence, stack_depth)
          if state == number then
              local num = torch.random(10)-1
              local op = prev_op_stack[stack_depth]
              local val = numerical_ans_stack[stack_depth]
              if     op == '+' then numerical_ans_stack[stack_depth] = val + num
              elseif op == '-' then numerical_ans_stack[stack_depth] = val - num
              else assert (false)
              end
              table.insert(ans, num)
              can_be_done = (num_open_parens == 0)
              num_numbers = num_numbers + 1
          elseif state == open_paren then
              table.insert(ans, '(')
              num_open_parens = num_open_parens + 1

              stack_depth = stack_depth + 1
              prev_op_stack[stack_depth] = '+'
              numerical_ans_stack[stack_depth] = 0
              can_be_done = false
          elseif state == closed_paren then
              if (stack_depth < 2) then
                  ------ this should basically never happen.
                  print ('state_sequence = ', state_sequence)
                  pbint ('stack_depth_sequence = ', stack_depth_sequence)
                  print ('ans = ', ans)
                  
              end
              assert (stack_depth >= 2)

              table.insert(ans, ')')
              num_open_parens = num_open_parens - 1

              stack_depth = stack_depth - 1
              local op = prev_op_stack[stack_depth]
              local val = numerical_ans_stack[stack_depth]
              local arg = numerical_ans_stack[stack_depth + 1]
              if op == '+' then val = val + arg
              else val = val - arg
              end 
              numerical_ans_stack[stack_depth] = val 
              can_be_done = (num_open_parens == 0) 
          elseif state == operator then
              local op = ({'+', '-'})[torch.random(2)]
              prev_op_stack[stack_depth] = op
              table.insert(ans, op)
              can_be_done = false
          end

          if len > max_len or stack_depth > max_depth or num_numbers >= max_nums then
              keep_going = false
          end
          local trans
          if keep_going then
              trans = state_trans
          else
              trans = done_state_trans
          end
          state = (trans[state])[torch.random(#trans[state])]

          ---- we don't want to go over the depth.
          if stack_depth == max_depth and state == open_paren then 
          ---- don't open new parens once you're deep enough.
              state = number
          end

          assert (state ~= nil)

          ---- do not close parens when it can no loner be done
          if num_open_parens == 0 and state == closed_paren then
              state = operator
          end
          if (state == closed_paren) then
              assert (num_open_parens >= 1) 
              assert (stack_depth >= 2)
          end

          if can_be_done and (keep_going == false) then
              done = true
          end

          len = len + 1
      end
      assert (stack_depth == 1)
      local numerical_ans = numerical_ans_stack[1]
      return ans, numerical_ans
   end

   --assert (num_syms >= 30)
   local input_tape_len = 3*absolute_max_len
   local input_tape = torch.zeros(batch_size, input_tape_len, num_syms)
   local output_tape = torch.zeros(batch_size, 6)
   local output_mask = torch.zeros(batch_size, 6)

   for b = 1, batch_size do
       local seq, ans = gen_one_example(b)
       while (#seq > input_tape_len) do
           seq, ans = gen_one_example()
       end
       local p = 0
       for pp = 1, #seq do
           p = p + 1
           local sym = visualize.inv_char_map[''..seq[p]]
           if sym == nil then
              print ('uh oh')
              print ('seq = ', seq, 'seq[p] = ', seq[p], 'p= ', p)
           end
           assert (sym > 0)
           assert (sym <= num_syms)
           input_tape[{b, p, sym}] = 1
       end
       p = p + 1
       input_tape[{b, p, visualize.inv_char_map['=']}] = 1
       for pp = #seq + 2, input_tape_len do
           p = p + 1
           input_tape[{b, p, visualize.inv_char_map[' ']}] = 1
       end
       local ans_spell = ''..ans
       local num_digits = #ans_spell
       assert (num_digits <= output_mask:size(2))
       local pos = 1
       local start = 1
       for d = start, num_digits do
           output_tape[{b, pos}] = visualize.inv_char_map[string.sub(ans_spell, d, d)]
           output_mask[{b, pos}] = 1
           pos = pos + 1
       end
       output_tape[{b, pos}] = visualize.inv_char_map['.']
       output_mask[{b, pos}] = 1
       pos = pos + 1
       while pos <= output_mask:size(2) do
           output_tape[{b, pos}] = 1
           output_mask[{b, pos}] = 0
           pos = pos + 1
       end 
       


   end 
   return input_tape, output_tape, output_mask
end


tasks.copy_curriculum = {}
function tasks.copy(len, batch_size, num_syms)
    local input_tape = torch.zeros(batch_size, len, num_syms)
    local output_tape = torch.zeros(batch_size, len):random(num_syms)
    local output_mask = torch.zeros(batch_size, len):fill(1)
    for b = 1, batch_size do
        for l = 1,len do
            input_tape[{b, l, output_tape[{b,l}] }] = 1
        end
    end
    return input_tape, output_tape, output_mask
end


--function tasks.rev(len, batch_size, num_syms)
--    local input_tape = torch.zeros(batch_size, len, num_syms)
--    local output_tape = torch.zeros(batch_size, len):random(num_syms)
--    local output_mask = torch.zeros(batch_size, len):fill(1)
--    for b = 1, batch_size do
--        for l = 1,len do
--            input_tape[{b, len-l+1, output_tape[{b,l}] }] = 1
--        end
--    end
--    return input_tape, output_tape, output_mask
--end

tasks.rev_curriculum = { {} }
function tasks.rev(len, batch_size, num_syms)
    local input_tape = torch.zeros(batch_size, len, num_syms)
    local len2 = math.ceil(len / 2)
    input_tape[{{}, {}, 1}]:fill(1)
    local output_tape = torch.zeros(batch_size, len2):random(num_syms-1):add(1)
    local output_mask = torch.zeros(batch_size, len2):fill(1)
    for b = 1, batch_size do
        for l = 1,len2 do
            input_tape[{b, len2-l+1}]:fill(0)
            input_tape[{b, len2-l+1, output_tape[{b,l}] }] = 1
        end
    end
    return input_tape, output_tape, output_mask
end

tasks.revCurriculum_curriculum = {}
for i = 1, 40 do
    tasks.revCurriculum_curriculum[i] = i
end
tasks.revCurriculum_hardcurriculum = {[1] = 100}

function tasks.revCurriculum(len, batch_size, num_syms, difficulties)
    assert (len >= 7)
    local input_tape = torch.zeros(batch_size, len, num_syms)
    local output_tape = torch.zeros(batch_size, len):fill(1)
    local output_mask = torch.zeros(batch_size, len)
    for b = 1, batch_size do
        --- local len_b = torch.random(math.ceil(len/2))
        local len_b = difficulties[b]
	if batch_size == 2 and b == 1 then
	   print ('revCurriculum: difficulty (len_b) = ', len_b)
	   print ('but len = ', len)
	end
        if len_b > len/2 then len_b = len/2 end
        for l = 1,len_b do
            local sym = torch.random(num_syms - 1) + 1
	    input_tape[{b, len_b-l+1, sym}] = 1
            output_tape[{b, l}] = sym            
            output_mask[{b, l}] = 1
        end
	for l = len_b+1, len do
	   local sym = visualize.inv_char_map[' ']
	   input_tape[{b, l, sym}] = 1
	end
        output_tape[{b, len_b + 1}] = 1
        output_mask[{b, len_b + 1}] = 1
    end
    return input_tape, output_tape, output_mask
end


tasks.repeatCopy_curriculum = {}
local p = 1
for rep = 2, 2 do
   for len = 1, 30 do
      tasks.repeatCopy_curriculum[p] = {}
      tasks.repeatCopy_curriculum[p].cur_len = len
      tasks.repeatCopy_curriculum[p].num_repeats = rep
      p = p + 1
   end
end

tasks.repeatCopy_hardcurriculum = {[1] = {cur_len = 30, num_repeats = 3}}
function tasks.repeatCopy(max_len, batch_size, num_syms, difficulties)
   local input_tape = torch.zeros(batch_size, max_len*3, num_syms):fill(0)
   local max_output_factor = 5
   local output_tape = torch.zeros(batch_size, max_output_factor*max_len):fill(1)
   local output_mask = torch.zeros(batch_size, max_output_factor*max_len)
   
   for b = 1, batch_size do
      local phrase_len = difficulties[b].cur_len
      if phrase_len > max_len then
	 phrase_len = max_len 
      end

      local num_repeats = difficulties[b].num_repeats
      if num_repeats > max_output_factor then
	 num_repeats = max_output_factor
      end

--      output_tape[{b, 1}] = visualize.inv_char_map['A']
--      output_mask[{b, 1}] = 1

      for l = 1, phrase_len do
	 local sym
	 if l < phrase_len then
	    sym = torch.random(num_syms)
	 else
	    sym = visualize.inv_char_map['.']
	 end
	 input_tape[{b, l, sym}] = 1
	 for c = 1, num_repeats  do
	    output_tape[{b, (c-1)*phrase_len + l}] = sym
	    output_mask[{b, (c-1)*phrase_len + l}] = 1
	 end
      end
      for l = phrase_len + 1, input_tape:size(2) do
	 local sym = visualize.inv_char_map[' ']
	 input_tape[{b, l, sym}] = 1
      end
   end
   return input_tape, output_tape, output_mask
end








tasks.repeatCopyN_curriculum = {}
local p = 1
for rep = 2, 3 do ---- keep it gentle: don't go over 3.
   for len = 1, 30 do
      tasks.repeatCopyN_curriculum[p] = {}
      tasks.repeatCopyN_curriculum[p].cur_len = len
      tasks.repeatCopyN_curriculum[p].num_repeats = rep
      p = p + 1
   end
end

function tasks.repeatCopyN(max_len, batch_size, num_syms, difficulties)
   local input_tape = torch.zeros(batch_size, max_len*3, num_syms):fill(0)
   local max_output_factor = 4
   local output_tape = torch.zeros(batch_size, max_output_factor*max_len + 1):fill(1)
   local output_mask = torch.zeros(batch_size, max_output_factor*max_len + 1)
   
   for b = 1, batch_size do
      local phrase_len = difficulties[b].cur_len
      if phrase_len > max_len then
	 phrase_len = max_len 
      end

      local num_repeats = difficulties[b].num_repeats
      if num_repeats > max_output_factor then
	 num_repeats = max_output_factor
      end

      input_tape[{b, 1, num_repeats}] = 1
      for l = 1, phrase_len do
	 local sym
	 if l < phrase_len then
	    sym = torch.random(num_syms - max_output_factor) + max_output_factor
	 else
	    sym = visualize.inv_char_map['.']
	 end
	 input_tape[{b, l+1, sym}] = 1
	 for c = 1, num_repeats  do
	    output_tape[{b, (c-1)*phrase_len + l}] = sym
	    output_mask[{b, (c-1)*phrase_len + l}] = 1
	 end
      end
      for l = phrase_len + 1, input_tape:size(2)-1 do
	 local sym = visualize.inv_char_map[' ']
	 input_tape[{b, l + 1, sym}] = 1
      end
   end
   return input_tape, output_tape, output_mask
end
----- next step:  sort 10 numbers.  Do it, now.  It should be doable in 45 steps.  Let's make it happen, damn it.


tasks.sort_curriculum = {}
for len = 1, 30 do
   table.insert(tasks.sort_curriculum, len)
end

function tasks.sort(max_len, batch_size, num_syms, difficulties)
   local input_tape = torch.zeros(batch_size, max_len, num_syms)
   local output_tape = torch.zeros(batch_size, max_len):fill(1)
   local output_mask = torch.zeros(batch_size, max_len)
   for b = 1, batch_size do
      local num_nums = difficulties[b]
      if num_nums > max_len - 3 then
	 num_nums = max_len - 3
      end
      local numbers = {}
      for t = 1, num_nums do
	 ---- you don't want that dot in the midsequence.
	 numbers[t] = visualize.inv_char_map['.'] 
	 while numbers[t] == visualize.inv_char_map['.'] do
	    numbers[t] = torch.random(num_syms)
	 end
	 assert (numbers[t] ~= visualize.inv_char_map['.'])
	 input_tape[{b, t, numbers[t]}] = 1
      end
      input_tape[{b, num_nums + 1, visualize.inv_char_map['.']}] = 1
      for t = num_nums + 2, max_len do
	 input_tape[{b, t, visualize.inv_char_map[' ']}] = 1
      end
      table.sort(numbers)
      for t = 1, num_nums do
	 output_tape[{b, t}] = numbers[t]
	 output_mask[{b, t}] = 1
      end
      output_tape[{b, num_nums + 1}] = visualize.inv_char_map['.']
      output_mask[{b, num_nums + 1}] = 1
   end
   return input_tape, output_tape, output_mask
end


tasks.sortB_curriculum = {}
for len = 1, 30 do
   tasks.sortB_curriculum[len] = len
end

function tasks.sortB(max_len, batch_size, num_syms, difficulties)
   local input_tape = torch.zeros(batch_size, max_len, num_syms)
   local output_tape = torch.zeros(batch_size, max_len):fill(1)
   local output_mask = torch.zeros(batch_size, max_len)
   for b = 1, batch_size do
      local num_nums = difficulties[b]
      if num_nums > max_len - 3 then
	 num_nums = max_len - 3
      end
      local numbers = {}
      for t = 1, num_nums do
	 ---- you don't want that dot in the midsequence.
	 numbers[t] = visualize.inv_char_map['.'] 
	 while numbers[t] == visualize.inv_char_map['.'] do
	    numbers[t] = torch.random(10)
	 end
	 assert (numbers[t] ~= visualize.inv_char_map['.'])
	 input_tape[{b, t, numbers[t]}] = 1
      end
      input_tape[{b, num_nums + 1, visualize.inv_char_map['.']}] = 1
      for t = num_nums + 2, max_len do
	 input_tape[{b, t, visualize.inv_char_map[' ']}] = 1
      end
      table.sort(numbers)
      for t = 1, num_nums do
	 output_tape[{b, t}] = numbers[t]
	 output_mask[{b, t}] = 1
      end
      output_tape[{b, num_nums + 1}] = visualize.inv_char_map['.']
      output_mask[{b, num_nums + 1}] = 1
   end
   return input_tape, output_tape, output_mask
end



function tasks.simpleAddition(max_len, batch_size, num_syms, difficulties)

end




function tasks.repeatInput(len, batch_size, num_syms, rep) --- this is a notion of difficulty.  Right.   It would be good to restructure it. 
    local input_tape = torch.zeros(batch_size, len*rep, num_syms)
    local output_tape = torch.zeros(batch_size, len)
    local output_mask = torch.zeros(batch_size, len):fill(1)
    for b = 1, batch_size do
        local pos = 0
        for l = 1,len do
            local sym = torch.random(num_syms)
            output_tape[{b, l}] = sym
            for r = 1, rep do
                pos = pos + 1
                input_tape[{b, pos, sym}] = 1
            end
        end
    end
    return input_tape, output_tape, output_mask
end

tasks.repeatInputCurriculum_curriculum = {[1] = {rep = 3, len = 0}}
tasks.repeatInputCurriculum_hardcurriculum = {[1] = {rep = 3, len = 1}}
function tasks.repeatInputCurriculum(len, batch_size, num_syms, difficulties)
   local rep = difficulties[1].rep
   local input_tape = torch.zeros(batch_size, len*rep+1, num_syms)
   local output_tape = torch.zeros(batch_size, len+1):fill(1) --- 1 = End of Sequence
   local output_mask = torch.zeros(batch_size, len+1)
   for b = 1, batch_size do
        local pos = 0

        local len_b
	if difficulties[1].len == 0 then
	   len_b = torch.random(len-1)+1
	else
	   len_b = len
	end

        assert (len_b >= 2)
        for l = 1,len_b do
            local sym = torch.random(num_syms-1)+1
            output_tape[{b, l}] = sym
            output_mask[{b, l}] = 1
            for r = 1, rep do
                pos = pos + 1
                input_tape[{b, pos, sym}] = 1 
            end
        end
        pos = pos + 1
        local end_sym = visualize.inv_char_map[' ']
        input_tape[{b, pos, end_sym}] = 1
        output_tape[{b, len_b+1}] = 1 --- end of sequence marker.
        output_mask[{b, len_b+1}] = 1 
    end
    return input_tape, output_tape, output_mask
end


return tasks
