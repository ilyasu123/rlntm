
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

require 'nn'
utils_2 = require 'utils_2'
require 'nngraph'
require 'rlntm_pkgs.Embedding'
local utils = require 'rlntm_pkgs.utils'

local misc = {}


function misc.concat(...)
   local tables = {...}
   local ans = {}
   for _, t in ipairs(tables) do 
       for _, e in ipairs(t) do
           table.insert(ans, e)
       end
   end
   return ans
end 

function misc.same_size(size1, size2)
   if #size1 ~= #size2 then
       return false
   end
   local nDimension = #size1
   for i = 1, nDimension do
       if size1[i] ~= size2[i] then
           return false
       end              
   end
   return true
end

function misc.sum(t)
   local ans = 0
   for i, v in ipairs(t) do
      ans = ans + v
   end
   return ans
end

function misc.prod(t)
   local ans = 1
   for i, v in ipairs(t) do
      ans = ans * v
   end
   return ans
end

function misc.convert(x)
   --if USE_GPU then 
   --  return x:cuda()
   --else 
   --   return x
   --end
   return x
end

local matrices = {}
function misc.named_zeros(name, ...)
   local size = {...}
   if size.totable ~= nil then 
      size = size:totable() 
   elseif size[1] ~= nil and type(size[1]) ~= 'number' then 
      if size[1].totable ~= nil then 
	 size = size[1]:totable() 
      end
   end
   
   if matrices[name] == nil then
      matrices[name] = misc.convert(torch.zeros(unpack(size)))
   end
   if matrices[name]:nElement() ~= misc.prod(size) then
      print ('warning:')
      print ('name = ', name, 'size = ', size)
      print ('matrices[', name, ']:size = ', unpack(matrices[name]:size():totable()))
      print ('size = ', unpack(size))
      print ('the two don\'t match.')
   end

   return matrices[name]:resize(torch.LongStorage(size))
end



function misc.clone_zero(x)
    if utils.istensor(x) then
        return torch.zeros(x:size())
    elseif utils.istable(x) then
        local ans =  {}
        for k,v in pairs(x) do
            ans[k] = misc.clone_zero(v)
        end
        return ans
    else
        assert(false, string.format("misc.clone_zero: can only clone tables or/of tensors -- it is %s", x))
    end
end

local __lin_map = {}
function misc.make_lin(sig, a, b, init)
    if __lin_map[sig] == nil then
        __lin_map[sig] = nn.Linear(a, b)
        if init ~= nil then
            if init.val ~= nil then
                __lin_map[sig].weight:fill(init.val)
                __lin_map[sig].bias:fill(init.val)
            end
        end
    end
    local ans = utils_2.cloneManyTimes(__lin_map[sig], 1)[1]
    ans.name = sig
    return ans
end

local __emb_map = {}
function misc.make_emb(sig, a, b, init)
    if __emb_map[sig] == nil then
        __emb_map[sig] = nn.Embedding(a, b, sig)
        if init ~= nil then
            if init.val ~= nil then
                __emb_map[sig].weight:fill(init.val)
            end
        end
    end
    local ans = utils_2.cloneManyTimes(__emb_map[sig], 1)[1]
    ans.name = sig
    return ans
end

function misc.inner_product(a, b)
   if type(a)=="number" then
       assert(type(b)~="number")
       return b*a
   elseif type(b) == "number" then
       assert(type(a)~="number")
       return a*b
   elseif utils.istensor(a) then
       assert(utils.istensor(b))
       return a:dot(b)
   else
       assert(utils.istable(a))
       assert(utils.istable(b))
       assert (#a == #b)
       local ans = 0
       for k = 1,#a do
           ans = ans + misc.inner_product(a[k], b[k])
       end
       return ans
   end
end

function misc.find_module_by_param(core_network, i)
  local start = 0
  local done = false
  local ret = ""
  local function size(param)
    if param == nil then
      return 0
    end
    if param:dim() == 2 then
      return param:size(1) * param:size(2)
    elseif param:dim() == 1 then
      return param:size(1)
    else
      assert(0)
    end
  end
  local function get_params(node)
    start = start + size(node.weight)
    start = start + size(node.bias)
    if not done then
      if start >= i then
        ret = node.__typename
        if node.name ~= nil then
          ret = string.format("%s,%s", ret, node.name)
        end
        done = true
        ret = {ret, node.info}
        return
      end
    end
  end
  core_network:apply(get_params)
  return unpack(ret)
end

function misc.inv_table(tab)
    local ans = {}
    for k,v in pairs(tab) do
        ans[v] = k
    end
    return ans
end



return misc
