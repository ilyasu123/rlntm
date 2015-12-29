
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

local check_grad_gmodule = {}
local utils = require 'utils'
local misc = require 'misc'

function check_grad_gmodule.check(gmodule, input, gradOutput, x, dx, eps)
    assert (utils.istable(input))
    dx:zero()
    local gx = dx:clone()
    gmodule:forward(input)
    gmodule:backward(input, gradOutput)
    for i = 1, dx:nElement() do
         local x_i = x[i]
         x[i] = x_i + eps
         local l1 = misc.inner_product(gmodule:forward(input), gradOutput)
         x[i] = x_i - eps
         local l2 = misc.inner_product(gmodule:forward(input), gradOutput)
         gx[i] = (l1 - l2) / (2*eps)
    end
    local err = gx:clone():add(-1,dx):norm() / dx:norm()
    return err, gx, dx
end

return check_grad_gmodule
