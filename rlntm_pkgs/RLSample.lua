
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

local misc = require 'rlntm_pkgs.misc'
local RLSample, parent = torch.class('nn.RLSample','nn.Module')
function RLSample:__init(action_map)
    parent.__init(self)
    self.action_map = action_map
end

function RLSample:updateOutput(input)
    assert (#input == 2)
    local action_dist, baseline = unpack(input)
    local batch_size, num_possible_actions = unpack(action_dist:size():totable())
    assert (num_possible_actions == #self.action_map)
    assert (baseline:nDimension() == 1 or (baseline:nDimension() == 2 and baseline:size(2) == 1))
    assert (baseline:nElement() == batch_size)

    ----- step 1:  sample this shit. 
    self.probs = (self.probs or torch.zeros(action_dist:size())):resize(action_dist:size())
    self.cumprobs = (self.cumprobs or torch.zeros(action_dist:size())):resize(action_dist:size())
    self.probs:exp(action_dist)

    self.cumprobs:cumsum(self.probs, 2)

    local rnd = torch.rand(batch_size,1):expandAs(self.cumprobs)

    local indication = self.cumprobs:le(rnd)

    self.sample = torch.sum(indication, 2):add(1):view(batch_size)

    --print ('RLSample: probs = ', self.probs, self.sample)
    
    self.output:resize(self.sample:size())
    for b = 1, batch_size do
         self.output[b] = self.action_map[self.sample[b]]
    end
    return self.output
end

function RLSample:updateGradInput(input, gradOutput)
   local action_dist, baseline = unpack(input)
   local batch_size, num_possible_actions = unpack(action_dist:size():totable())
   gradOutput = gradOutput:view(batch_size,1)
   baseline = baseline:view(batch_size,1)
   assert (misc.same_size(gradOutput:size(), baseline:size()))
   local d_action_dist = torch.zeros(action_dist:size())
   local d_baseline = torch.add(gradOutput, -1, baseline)
   for b = 1, batch_size do
       if math.abs(gradOutput[{b,1}]) > 1e-4 then
          d_action_dist[{b, self.sample[b]}] = d_baseline[{b,1}]    --- give this derivative to the lucky sampled action.
       else
          d_baseline[{b,1}] = 0
       end
   end 
   self.gradInput = {d_action_dist, d_baseline}
   return self.gradInput
end
