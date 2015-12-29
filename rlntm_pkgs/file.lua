
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--


local ff = {}

function ff.Exists(fname)
   local f=io.open(fname,"r")
   if f~=nil then io.close(f) return true else return false end
end

function ff.GetContents(fname)
   local file = assert(io.open(fname))
   local contents = file:read '*a'
   return
end

function ff.SetContents(fname, txt)
   os.execute("rm -rf " .. fname)
   local file = assert(io.open(fname, "w"))
   file:write(txt)
   file:close()
end

function ff.MakeDir(dir)
   os.execute("mkdir " .. dir)
end

return ff