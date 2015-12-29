
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

local Embedding, parent = torch.class('nn.Embedding', 'nn.Module')

function Embedding:__init(inputSize, outputSize, name)
  parent.__init(self)
  self.outputSize = outputSize
  self.weight = torch.zeros(inputSize, outputSize)
  self.gradWeight = torch.zeros(inputSize, outputSize)
  self.name = name
  print(string.format("Creating embeeding %s with inputSize = %d", name, inputSize))
end

function Embedding:updateOutput(input)
  --dprint(string.format("FP Embedding = %s, input:size(1) = %d", self.name, input:size(1)))
  assert(input:nDimension() == 1 or input:size(2) == 1)
  self.output:resize(input:size(1), self.outputSize):zero()
  for i = 1, input:size(1) do
    if input[i] > self.weight:size(1) or input[i] < 0 then
      print("Error occured")
      print("Forward pass of Embeeding")
      print("input = ", input)
      print("i = ", i)
      print("input[i] = ", input[i])
      print("name = ", self.name)
      exit(0)
    end
    if input[i] ~= 0 then
      self.output[i]:copy(self.weight[input[i]])
    end
  end
  return self.output
end

function Embedding:updateGradInput(input, gradOutput)
  if self.gradInput then
    self.gradInput:resize(input:size()):zero()
    return self.gradInput
  end
end

function Embedding:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  if scale == 0 then
    self.gradWeight:zero()
  end
  for i = 1, input:size(1) do
    if input[i] > self.weight:size(1) or input[i] < 0 then
      print("Error occured")
      print("Backward pass of Embeeding")
      print("input = ", input)
      print("i = ", i)
      print("input[i] = ", input[i])
      print("name = ", self.name)
      exit(0)
    end
    if input[i] ~= 0 then
      self.gradWeight[input[i]]:add(gradOutput[i])
    end
  end
end

Embedding.sharedAccUpdateGradParameters = Embedding.accUpdateGradParameters
