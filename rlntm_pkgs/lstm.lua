
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

require 'nn'
require 'utils_2'
require 'nngraph'
local misc = require 'rlntm_pkgs.misc'

local lstm = {}

function lstm.core(sig, x,h,c, in_size, rnn_size, forget_gate_bias, init)

   if forget_gate_bias == nil then forget_gate_bias = 0 end
   assert (x ~= nil and h ~= nil and c ~= nil)
   local function input(sig2)
      return nn.CAddTable(){misc.make_lin(sig..'_'..sig2..'_x', in_size , rnn_size, init)(x),
                            misc.make_lin(sig..'_'..sig2..'_h', rnn_size, rnn_size, init)(h)}
   end
   local i1 = nn.Tanh   ()(input('i1'))
   local i2 = nn.Sigmoid()(input('i2'))
   local f  = nn.Sigmoid()(nn.ConstAdd(forget_gate_bias)(input('f' ))) --- lots of forgetting.
   local o  = nn.Sigmoid()(input('o' ))

   local ii = nn.CMulTable(){i1,i2}
   local cf = nn.CMulTable(){c, f}
   local c_new = nn.CAddTable(){cf, ii}
   local h_new = nn.CMulTable(){o, nn.Tanh()(c_new)}

   return h_new, c_new
end


function lstm.deep(depth, H_prev,  x, in_size, rnn_size, forget_gate_bias, sig_extra, init)
  if sig_extra == nil then sig_extra = '' end         
  H_prev = {H_prev:split(2*depth)}
  local H_cur = {} 
  local input = x
  rnn_size = rnn_size or in_size
  local cur_in_size = in_size
  local pos = 0
  for layer = 1, depth do
     local h_pos, c_pos = pos + 1, pos + 2
     pos = pos + 2
     local sig = string.format('LSTM: layer = %d', layer)
     H_cur[h_pos], H_cur[c_pos] = lstm.core(sig_extra .. sig, input, H_prev[h_pos], H_prev[c_pos], cur_in_size, rnn_size, forget_gate_bias, init)
     input = H_cur[h_pos]
     cur_in_size = rnn_size
  end
  assert (#H_cur == 2*depth)
  return input, nn.Identity()(H_cur)
end

return lstm
