
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

------------------------------------------------------------
------------------------------------------------------------
------------------------------------------------------------
------------------------------------------------------------
------- IMPORTANT NOTE
------------------------------------------------------------
------------------------------------------------------------
------------------------------------------------------------
-- efficient_mem uses a hack:  it uses a global variable
-- for the forward memory state, and a global variable
-- for the backward grad memory state. 
------------------------------------------------------------
-- Therefore, it is essential to call efficient_mem.prepare(init_mem_state) 
-- before starting out.  Don't forget to do that, or else this 
-- thing won't work at all.  In brief, this works as follows:

----- in the forward pass, we make local updates to the memory.

----- in the backward pass, we undo the local updates to the memory.  We also do local updates to the memory gradient.

require 'nn'
require 'nngraph'
require 'utils_2'
local misc = require 'rlntm_pkgs.misc'

----- "GLOBAL" VARIABELS  (although they are at least partly local)
local mem_buf = torch.zeros(1)
local grad_mem_buf = torch.zeros(1)

local efficient_mem = {}
function efficient_mem.prepare(init_mem)
    mem_buf:resize(init_mem:size()):copy(init_mem)
    grad_mem_buf:resize(init_mem:size()):zero()
end

local ReadMem, parent = torch.class('nn.ReadMem','nn.Module')
function ReadMem:__init()
    parent.__init(self)
end

function ReadMem:updateOutput(input)
   assert (#input == 2)
   local tape, ptr = unpack(input)
   assert (tape:nElement() == 1)
   local batch_size, len, dim = unpack(mem_buf:size():totable())

   assert (ptr:nDimension() == 1)      
   assert (ptr:nElement() == batch_size) 
   --   assert (ptr:min() >= 1)
   
   self.output:resize(batch_size, dim)
--   if SPEED_TEST == true then return self.output end

   for b = 1, batch_size do
       self.output[b]:copy(mem_buf[{b, ((ptr[b]-1) % len) + 1}])
   end 
   return self.output
end

function ReadMem:updateGradInput(input, gradOutput)
   local tape, ptr = unpack(input)
   assert (tape:nElement() == 1)
   local batch_size, len, dim = unpack(mem_buf:size():totable())
   local d_tape = torch.zeros(1)
   local d_ptr  
--   if SPEED_TEST == false then
      for b = 1, batch_size do
	 grad_mem_buf[{b, ((ptr[b]-1) % len) + 1}]:add(gradOutput[b]) --- *INCREMENT* the memory gradinet buffer, since it's a global variable.
      end 
--   end
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
    assert (tape:nElement() == 1)
    assert (mem_buf:nDimension() == 3)
    local batch_size, len, dim = unpack(mem_buf:size():totable())

--    print ('nev_val:size = ', unpack(new_val:size():totable()))
--    print ('dim = ', dim)
--    print ('mem_buf:size = ', unpack(mem_buf:size():totable()))


    assert (ptr:nElement() == batch_size and ptr:nDimension() == 1)
    assert (decay:nDimension() == 2 and decay:size(1) == batch_size and decay:size(2) == dim)
    assert (new_val:nDimension() == 2)
    assert (new_val:size(1) == batch_size)
    assert (new_val:size(2) == dim)
    
    
    self.old_val = (self.old_buf or torch.zeros(batch_size, dim)):resize(batch_size, dim)
--    if SPEED_TEST == false then
       for b = 1, batch_size do
	  local p = ((ptr[b]-1) % len) + 1   
	  self.old_val[b] = mem_buf[{b, p}] -- store the current mem value for efficient undoing.
	  mem_buf[{b, p}]:cmul(decay[b])
	  mem_buf[{b, p}]:add(new_val[b]) 
       end    
--    end
    self.output = torch.zeros(1)
    return self.output
end

function WriteMem:updateGradInput(input, gradOutput)
    local tape, ptr, decay, new_val = unpack(input)  -- ptr doesn't get any gradients.  Good.
    assert (tape:nElement() == 1)
    local batch_size, len, dim = unpack(mem_buf:size():totable())
    assert (gradOutput:nElement() == 1)
    local d_decay = torch.zeros(decay:size())
    local d_new_val = torch.zeros(new_val:size())

    local d_ptr
    if mode == 'hack' then 
        d_ptr = 'nil'
    else
        d_ptr = torch.zeros(ptr:size())
    end
    ----- Undo the memory write.
--    if SPEED_TEST == false then
       for b = 1, batch_size do
	  local p = ((ptr[b]-1) % len) + 1   
	  mem_buf[{b, p}] = self.old_val[b] --- revert to the old memory value
	  
	  local p = ((ptr[b]-1) % len) + 1           
	  d_new_val[b]:add(grad_mem_buf[{b, p}])
	  d_decay[b]:cmul(grad_mem_buf[{b, p}], mem_buf[{b, p}])
	  grad_mem_buf[{b, p}]:cmul(grad_mem_buf[{b, p}], decay[b])
       end
--    end
    local d_tape = torch.zeros(1)
    self.gradInput = {d_tape, d_ptr, d_decay, d_new_val}
    return self.gradInput
end


return efficient_mem
