----------------------------------------------------------------------
-- This script generates the global avg of the input image dataset
-- This gives better prediction for object recognition
--
-- 
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'pl'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'camera'
require 'image'
require 'nnx'
require 'ffmpeg'

----------------------------------------------------------------------

print(sys.COLORS.red .. '==> defining some tools')
function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

function getmeans()

   local MtestData={}
   torch.setdefaulttensortype('torch.FloatTensor')
   torch.setnumthreads(opt.threads)


   local desImaX = 46 -- desired cropped dataset image size
   local desImaY = 46
   local cropTrX = 45 -- desired offset to crop images from train set
   local cropTrY = 48

   local cropTeX = 33 -- desired offset to crop images from test set
   local cropTeY = 35
   local testDir = 'INRIAPerson/70X134H96/Test/pos/'
   local testImaNumber = #ls(testDir)
   local labelPerson = 1
   local labelBg = 2
   totSize = (testImaNumber-1)*2 -- twice because of bg data!
   local dspath = 'INRIAPerson/bg.m4v'
   local source = ffmpeg.Video{path=dspath, width=284/2, height=160/2, encoding='png', 
            fps=30, lenght=100, delete=false, load=false}

   local rawFrame = source:forward()

   ivch = rawFrame:size(1) -- channels
   ivhe = rawFrame:size(2) -- height
   ivwi = rawFrame:size(3) -- width
   --print("sizes")
   print(ivch)
   print(ivhe)
   print(ivwi)

   MtestData = {
      data = torch.Tensor(totSize, ivch,desImaX,desImaY),
      labels = torch.Tensor(totSize),
      size = function() return totSize end
   }
   --print(MtestData)
   --names={}
   -- load person test data: 1126 images
   for i = 1, totSize, 2 do
      img = image.loadPNG(testDir..ls(testDir)[(i-1)/2+1],ivch) -- we pick all of the images in test!
      --table.insert(names,ls(testDir)[(i-1)/2+1])
      MtestData.data[i] = image.crop(img, cropTeX-desImaX/2, cropTeY-desImaY/2, 
      cropTeX+desImaX/2, cropTeY+desImaY/2):clone()
      MtestData.labels[i] = labelPerson

      -- load background data:
      
      img = source:forward()
      local x = math.random(1,ivwi-desImaX+1)
      local y = math.random(15,ivhe-desImaY+1-30) -- added # to get samples more or less from horizon
      MtestData.data[i+1] = img[{ {},{y,y+desImaY-1},{x,x+desImaX-1} }]:clone()
      MtestData.labels[i+1] = labelBg
   end
      

   --torch.save('ntest.t7',MtestData)   
   -------------------------------------------------------------

   local channels = {'r','g','b'}

   -- Normalize each channel, and store mean/std
   -- per channel. These values are important, as they are part of
   -- the trainable parameters. At test time, test data will be normalized
   -- using these values.
   print(sys.COLORS.red ..  '==> preprocessing data: global normalization:')
   local mean = {}
   local std = {}
   size=MtestData.data:size(1)
   --print(size)
   -- Normalize test data, using the training means/stds
   for i,channel in ipairs(channels) do
      -- normalize each channel globally:
      mean[i] = MtestData.data[{ {},i,{},{} }]:mean()
      std[i] = MtestData.data[{ {},i,{},{} }]:std()
   end

   return mean,std
end