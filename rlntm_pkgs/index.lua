
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

require 'nn'

local Index, parent = torch.class('nn.MyIndex','nn.Module')
function Index:__init(dim)
     self.dim = dim
     parent.__init(self)
end

function Index:updateOutput(input)
     assert(#input == 2)
     local mat, inds = unpack(input)
     assert (mat:nDimension() == 2)
     local batch_size, dim = unpack(mat:size():totable())
     self.output:resize(batch_size)
--     if SPEED_TEST == false then
	assert (inds:max() <= dim)
	assert (inds:min() >= 1)
	assert (inds:nElement() == batch_size and inds:nDimension() == 1)
	for b = 1, batch_size do
	   self.output[b] = mat[{b, inds[b]}]
	end
--     end
     return self.output
end


function Index:updateGradInput(input, gradOutput)
     local mat, inds = unpack(input)
     local batch_size, dim = unpack(mat:size():totable())
     self.d_mat = (self.d_mat or torch.Tensor(mat:size())):zero()
--     if SPEED_TEST == false then
	for b = 1, batch_size do
	   self.d_mat[{b, inds[b]}] = gradOutput[b]
	end
--     end
     local d_inds
     if mode == 'hack' 
     then d_inds = "nil"
     else d_inds = torch.zeros(inds:size())
     end
     self.gradOutput = {self.d_mat, d_inds}
     return self.gradOutput
end
