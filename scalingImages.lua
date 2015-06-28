---------------------------------------------------------------------
-- This script tries to consider only those images with person 
-- It generates images of different scale
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

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining some tools')

function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!


local testData={}
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(8)



local desImaX = 46 -- desired cropped dataset image size
local desImaY = 46
local cropTrX = 45 -- desired offset to crop images from train set
local cropTrY = 48

local cropTeX = 33 -- desired offset to crop images from test set
local cropTeY = 35

local testDir = 'INRIAPerson/70X134H96/Test/pos/'
--local testDir = 'scratch/bg.m4v_30fps_142x80_10s_c0_sk0_png/'
local testImaNumber = #ls(testDir)
local labelPerson = 1
local labelBg = 2



print(teSize)
ivch = 3 -- channels
positions={}


names={}
local testImaNumber = #ls(testDir)
local labelPerson = 1
local labelBg = 2
local totSize = (testImaNumber-1) -- twice because of bg data!
positionSize=0
print(totSize)
index=0
scale={0.5,0.6,0.8}
scSize=3
xSize=0
ySize=0
-- load person test data: 1126 images
for k = 1, totSize, 1 do
  img = image.loadPNG(testDir..ls(testDir)[k],ivch) -- we pick all of the images in test!
  size=img:size()
  print(size)
  teSize = math.floor((size[2]-47)*(size[3]-47)/100) -- twice because of bg data!
  for j=1,3,1 do  -- gives error if 46

     xSize=math.floor(scale[j]*size[3])
     ySize=math.floor(scale[j]*size[2])
      positionSize = positionSize  + 1
      index=index+1
      print(positionSize)
      local newImage = image.scale(img,xSize,ySize):clone()
      tfile = 'scaled/'.. tostring(index) .. '.png'
      image.save(tfile,newImage)
      print(tfile)

            --print(testData.data[index]:size())
            --print(i,j,i+46,j+46)
            
        --i1=i[{ {3},{1,46},{1,46} }] 
        
     
     
  end
  print(index)
end
