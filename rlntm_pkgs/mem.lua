
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

require 'nn'
require 'nngraph'
utils_2 = require 'utils_2'
local misc = require 'rlntm_pkgs.misc'

local ReadMem, parent = torch.class('nn.ReadMem','nn.Module')
function ReadMem:__init()
    parent.__init(self)
end

function ReadMem:updateOutput(input)
   assert (#input == 2)
   local tape, ptr = unpack(input)
   assert (tape:nDimension() == 3)
   local batch_size, len, dim = unpack(tape:size():totable())

   assert (ptr:nDimension() == 1)      
   assert (ptr:nElement() == batch_size) 
   --   assert (ptr:min() >= 1)
   

   self.output:resize(batch_size, dim)
   for b = 1, batch_size do
       self.output[b]:copy(tape[{b, ((ptr[b]-1) % len) + 1}])
   end 
   return self.output
end

function ReadMem:updateGradInput(input, gradOutput)
   local tape, ptr = unpack(input)
   local batch_size, len, dim = unpack(tape:size():totable())
   local d_tape = torch.zeros(tape:size())
   local d_ptr  
   for b = 1, batch_size do
       --self.output[b]:copy(tape[{b, ((ptr[b]-1) % len) + 1}])
       d_tape[{b, ((ptr[b]-1) % len) + 1}]:copy(gradOutput[b])
   end 
   if mode == "hack" then
       d_ptr = "nil"
   else
       d_ptr = torch.zeros(ptr:size())
   end
   self.gradInput = {d_tape, d_ptr}
   return self.gradInput
end


local WriteMem, parent = torch.class('nn.WriteMem','nn.Module')
function WriteMem:__init()
    parent.__init(self)
end

function WriteMem:updateOutput(input)
    assert (#input == 4)
    local tape, ptr, decay, new_val = unpack(input)  -- ptr doesn't get any gradients.  Good.
    assert (tape ~= nil) 
    assert (ptr ~= nil)   
    assert (tape:nDimension() == 3)
    local batch_size, len, dim = unpack(tape:size():totable())
    assert (ptr:nElement() == batch_size and ptr:nDimension() == 1)
    assert (decay:nDimension() == 2 and decay:size(1) == batch_size and decay:size(2) == dim)
    assert (new_val:nDimension() == 2 and new_val:size(1) == batch_size and new_val:size(2) == dim)
    
    local new_tape = tape:clone()
    for b = 1, batch_size do
        local p = ((ptr[b]-1) % len) + 1   
        new_tape[{b, p}]:cmul(decay[b])
        new_tape[{b, p}]:add(new_val[b]) 
    end    
    self.output = new_tape
    return self.output
end

function WriteMem:updateGradInput(input, gradOutput)
    local tape, ptr, decay, new_val = unpack(input)  -- ptr doesn't get any gradients.  Good.
    local batch_size, len, dim = unpack(tape:size():totable())
    local d_tape = gradOutput:clone()
    local d_decay = torch.zeros(decay:size())
    local d_new_val = torch.zeros(new_val:size())
    local d_ptr
    if mode == 'hack' then 
        d_ptr = 'nil'
    else
        d_ptr = torch.zeros(ptr:size())
    end
    
    for b = 1, batch_size do
        local p = ((ptr[b]-1) % len) + 1           
        d_new_val[b]:add(gradOutput[{b, p}])
        d_decay[b]:cmul(gradOutput[{b, p}], tape[{b, p}])
        d_tape[{b, p}]:cmul(gradOutput[{b, p}], decay[b])
    end
    self.gradInput = {d_tape, d_ptr, d_decay, d_new_val}
    return self.gradInput
end


