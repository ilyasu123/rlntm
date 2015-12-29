
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

require 'nn'
require 'nngraph'
require 'utils_2'
local misc = require 'rlntm_pkgs.misc'
local utils = require 'utils'
local GetInput, parent = torch.class('nn.GetInput','nn.Module')

function GetInput:__init(noloop)
   if noloop == nil then
      self.noloop = false
   else
      self.noloop = true
   end
   parent.__init(self)
end
function GetInput:updateOutput(input)
   ---- it's easy to move the input 

   assert (#input == 2)
   local tape, ptr = unpack(input)

   

   assert (tape ~= nil) 
   assert (ptr ~= nil)   
   assert (tape:nDimension() >= 2)
   local batch_size, len, _ = unpack(tape:size():totable())

   assert (ptr:nDimension() == 1)      
   assert (ptr:nElement() == batch_size) 


   local output_size = {batch_size}
   for d = 3, tape:nDimension() do
       table.insert(output_size, tape:size(d))
   end
   
   local function mapp(ptr)
       return  ((ptr-1) % len) + 1
   end

   self.output:resize(torch.LongStorage(output_size))
--   if SPEED_TEST == true then return self.output end

   if self.output:nDimension() > 1 then 
       for b = 1, batch_size do    
            self.output[b]:copy(tape[{b, mapp(ptr[b])}])
       end   
   else
       for b = 1, batch_size do    
            self.output[b] = tape[{b, mapp(ptr[b])}]
       end   
   end
   return self.output
end

function GetInput:updateGradInput(input, gradOutput)         
   if mode == "hack" then
       self.gradInput = "nil"
   else
       self.gradInput = misc.clone_zero(input)
   end
   return self.gradInput
end

--------------------------------------------------------------------------
local MovePtr, parent = torch.class('nn.MovePtr','nn.Module')
function MovePtr:__init()
    parent.__init(self)
end

function MovePtr:updateOutput(input)
   assert (#input == 2)
   local ptr, moves = unpack(input)
   self.output:resize(ptr:size())
   self.output:add(ptr, moves)
   return self.output 
end

function MovePtr:updateGradInput(input, gradOutput)
   if mode == "hack" then
       self.gradInput = "nil"
   else
       self.gradInput = misc.clone_zero(input)
   end
   return self.gradInput
end         


---------------------------------------------------------------------------------
---- Input action constraints:

local InputActionConstraints, parent = torch.class('nn.InputActionConstraints','nn.Module')

function InputActionConstraints:__init(action_map)
    parent.__init(self)
    self.action_map = action_map
    self.p1 = nil
    self.p0 = nil
    self.m1 = nil
    assert (#action_map == 3)
    for i,a in pairs(action_map) do
       if a == -1 then self.m1 = i end
       if a == 1 then self.p1 = i end
       if a == 0 then self.p0 = i end
    end
    assert (self.p1 ~= nil and self.p0 ~= nil and self.m1 ~= nil)
    assert (self.p1 ~= self.p0 and self.p0 ~= self.m1 and self.m1 ~= self.p1)
end

function InputActionConstraints:updateOutput(input)
    assert (#input == 2)
    local input_tape, input_pos = unpack(input)
    local batch_size, input_len, num_syms = unpack(input_tape:size():totable())
    assert (input_pos:size(1) == batch_size)
    self.output:resize(batch_size, #self.action_map):zero()
    local huge = 200
    local max_input_len = input_tape:size(2)
    for b = 1, batch_size do
       self.output[{b,self.m1}] = -huge --- can't go back
       if input_pos[b] >= max_input_len then
	  self.output[{b,self.p1}] = -huge ---- we can't go forward either.
       end
       assert (self.output[b]:max()==0)
    end
    return self.output
end

function InputActionConstraints:updateGradInput(input, gradOutput)
   if mode == "hack" then
       self.gradInput = "nil"
   else
       self.gradInput = misc.clone_zero(input)
   end
   return self.gradInput
end

