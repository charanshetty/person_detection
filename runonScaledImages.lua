----------------------------------------------------------------------
-- This script tries to considers all images
-- It uses cropping and then predicting for objects
-- has 3 different scales
-- removes the writing and reading part
--However the unscaling is not working fine so using the already existing resize method
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

function runOnScaledImages( opt,p )
  -- body

opt = lapp[[
   -n, --network  (default 'model.net')   path to networkimage.
   -s, --save      (default '')
   -t, --threads  (default 8)             number of threads
   -b, --batchSize (default 128)
]]
local testData={}
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)


local desImaX = 46 -- desired cropped dataset image size
local desImaY = 46
local cropTrX = 45 -- desired offset to crop images from train set
local cropTrY = 48

local cropTeX = 33 -- desired offset to crop images from test set
local cropTeY = 35
local testDir = 'pedestrian/'
--local testDir = 'scratch/bg.m4v_30fps_142x80_10s_c0_sk0_png/'
local testImaNumber = #ls(testDir)
local labelPerson = 1
local labelBg = 2
index=0
scale={0.5,0.2,0.4}
local totSize = 3
scSize=3
xSize=0
ySize=0
names={}
local channels = {'r','g','b'}
dofile 'getmeans.lua'
local mean={}
local std ={}
mean,std = getmeans()
-- model:
  local t = require 'model'
  --local model = t.model
  local model = torch.load(opt.network)
  local loss = t.loss
  classes = {'person','backg'}
  local confusion = optim.ConfusionMatrix(classes)

    
for k = 1, totSize, 1 do
  local tmp = nil
  local tmp2 =nil
  Oimg = image.loadPNG(testDir..ls(testDir)[p],ivch) -- we pick all of the images in test!
  size=Oimg:size()
  print("sizes",size)
  xSize=math.floor(scale[k]*size[3])
  ySize=math.floor(scale[k]*size[2])
  print(size[1],size[2],size[3])
  print(ls(testDir)[p])
  img = Oimg:clone()   -- or else img wouldn t be initialised
  img= image.scale(img,xSize,ySize):clone() 
  size=img:size()
  print("after scaling")
  print(size[1],size[2],size[3])  
  if(size[3]<48)then
    img = image.scale(Oimg,48,size[2]):clone()  
    size=img:size()
    print(size)
  end
  if((size[2]< 57) and (size[2]>= 57)) then
    teSize = math.floor((size[3]-47)/10) --   for it to be >1
  elseif((size[3] < 57) and (size[2]>= 57)) then
    teSize = math.floor((size[2]-47)/10) --   for it to be >1
  elseif((size[3] < 57) and (size[2]< 57)) then
    teSize = 1
  else
    teSize = math.floor((size[2]-47)*(size[3]-47)/100) 
  end
  print("number of blocks",teSize)
  ivch = 3 -- channels
  positions={}
  testData = {
    data = torch.Tensor(teSize, ivch,desImaX,desImaY),
    labels = torch.Tensor(teSize),
    size = function() return teSize end
  }
  print(testData)
  positionSize=0
  index=0
  -- load person test data: 1126 images
    for j=1,size[2]-47,10 do  -- gives error if 46
       for i=1,size[3] -47,10 do
       	if(index<teSize) then   -- why I did this?
              table.insert(names,ls(testDir)[p])
              table.insert(positions,{i=index,x1=i,y1=j,x2=i+46,y2=j+46})

              positionSize = positionSize  + 1
              index=index+1
              print(positionSize)
  			      testData.data[index] = image.crop(img, i, j, i+46, j+46)
              print("where")
              print(testData.data[index]:size())
              print(i,j,i+46,j+46)
              testData.labels[index] = labelPerson
          end       
       end
    end
    torch.save('Allscaledtest.t7',testData)   
    print(sys.COLORS.red ..  'Test Data:')
    print(testData.data:size(1))
    print()
    print(sys.COLORS.red ..  '==> preprocessing data')

    
          -- Normalize each channel, and store mean/std
    -- per channel. These values are important, as they are part of
    -- the trainable parameters. At test time, test data will be normalized
    -- using these values.
    print(sys.COLORS.red ..  '==> preprocessing data: global normalization:')
    local mean = {}
    local std = {}
    size=testData.data:size(1)
    print(size)
    -- Normalize test data, using the training means/stds
    
    for i,channel in ipairs(channels) do
  -- normalize each channel globally:
  	   mean[i] = testData.data[{ {},i,{},{} }]:mean()
  	   std[i] = testData.data[{ {},i,{},{} }]:std()
  	   print("means",mean[i])
  	   testData.data[{ {},i,{},{} }]:add(-mean[i])
  	   testData.data[{ {},i,{},{} }]:div(std[i])
    end
  ----------------------------------------------------------------------
  print(sys.COLORS.red ..  '==> verify statistics:')

  -- It's always good practice to verify that data is properly
  -- normalized.

  for i,channel in ipairs(channels) do

     local testMean = testData.data[{ {},i }]:mean()
     local testStd = testData.data[{ {},i }]:std()

     
     print('       test data, '..channel..'-channel, mean:                   ' .. testMean)
     print('       test data, '..channel..'-channel, standard deviation:     ' .. testStd)
  end

  --------------------------------------------------------------------
  testData.size = function() return teSize end

  
  -- Lo/gger:
  local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

  -- Batch test:
  local inputs = torch.Tensor(opt.batchSize,testData.data:size(2), 
           testData.data:size(3), testData.data:size(4)) -- get size from data
  local targets = torch.Tensor(opt.batchSize)


  ----------------------------------------------------------------------
  print(sys.COLORS.red .. '==> defining test procedure')

  -- test function

  -- local vars
  local time = sys.clock()

  -- test over test data
  print(sys.COLORS.red .. '==> testing on test set:')
  print(testData:size())
  for t = 1,testData:size(),opt.batchSize do
     -- disp progress
     xlua.progress(t, testData:size())

     -- batch fits?
     if (t + opt.batchSize - 1) > testData:size() then
        break
     end

     -- create mini batch
     local idx = 1
     for i = t,t+opt.batchSize-1 do
     	  
        inputs[idx] = testData.data[i]
        targets[idx] = testData.labels[i]

        idx = idx + 1
     end

     -- test sample
     local preds = model:forward(inputs)
     print(preds ,"predicted")
     print("difference btw 2",preds[1][1]-preds[1][2])
     -- confusion
     for i = 1,opt.batchSize do
        print((t-1)*opt.batchSize+i)
        --if not (((((preds[i])[1]>(preds[i])[2]) and (targets[i]==2)) or (((preds[i])[1]<(preds[i])[2]) and (targets[i]==1))) and preds[1][1]-preds[1][2] <12) then
        if (((((preds[i])[1]>(preds[i])[2]) and (targets[i]==1)) and preds[1][1]-preds[1][2] > 12)) then
          print('found')  -- mismatch when predicted value is smaller than that of other class         
          print(names[(t-1)*opt.batchSize+i])
          --print(positions[(t-1)*opt.batchSize+i].x1,positions[(t-1)*opt.batchSize+i].y1,
          	--positions[(t-1)*opt.batchSize+i].x2,positions[(t-1)*opt.batchSize+i].y2)      
          boundingBox = {}
          print("there")
          print(positions[(t-1)*opt.batchSize+i].x1, positions[(t-1)*opt.batchSize+i].y1, 
          positions[(t-1)*opt.batchSize+i].x2,positions[(t-1)*opt.batchSize+i].y2)
          table.insert(boundingBox, 
          {xMin = positions[(t-1)*opt.batchSize+i].x1, 
          xMax = positions[(t-1)*opt.batchSize+i].x2,
          yMin = positions[(t-1)*opt.batchSize+i].y1, 
          yMax = positions[(t-1)*opt.batchSize+i].y2})

          for _,b in ipairs(boundingBox) do
            tfile = 'Scaledbox/'.. tostring(ls(testDir)[p]) ..tostring((t-1)*opt.batchSize+i).. '.png'
            print("here")
            --b.xMin=(math.floor(b.xMin*(1-((1-scale[k])/2))))+1  -- this is to unsize to what was actually resized 
            --b.yMin=(math.floor(b.yMin*(1-((1-scale[k])/2))))+1  -- + 1 to get rid of zero error
            --if((math.floor(b.xMax*(1+((1-scale[k])/2)))) < Oimg:size(3)) then b.xMax=(math.floor(b.xMax*(1+((1-scale[k])/2)))) end
            --if((math.floor(b.yMax*(1+((1-scale[k])/2)))) < Oimg:size(2)) then b.yMax=(math.floor(b.yMax*(1+((1-scale[k])/2)))) end
            print("afterUnsizing",b.xMin,b.yMin,b.xMax,b.yMax)
            if not(tmp2) then
              tmp = img:clone()
              Oimg = img:clone()
              Oimg[{ 2,b.yMin,{b.xMin,b.xMax} }] = 1
              Oimg[{ 2,b.yMax,{b.xMin,b.xMax}}] = 1
              Oimg[{ 2,{b.yMin,b.yMax}, b.xMin}] = 1
              Oimg[{ 2,{b.yMin,b.yMax}, b.xMax}] = 1
              image.save(tfile,Oimg)
              tmp2=Oimg:clone()
              img = tmp:clone()
              else
              tmp2[{ 2,b.yMin,{b.xMin,b.xMax} }] = 1
              tmp2[{ 2,b.yMax,{b.xMin,b.xMax}}] = 1
              tmp2[{ 2,{b.yMin,b.yMax}, b.xMin}] = 1
              tmp2[{ 2,{b.yMin,b.yMax}, b.xMax}] = 1
              print(tfile, created)
              image.save(tfile,tmp2)
              img = tmp:clone()
            end
          end
  		  end
        print(targets[i] ,"targetted") -- the most positive is the predicted class
        --print(i)
        confusion:add(preds[i], 1)
        print(confusion)
     end
  end

  -- timing
  time = sys.clock() - time
  time = time / testData:size()
  -- print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix


  -- update log/plot
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  if opt.plot then
     testLogger:style{['% mean class accuracy (test set)'] = '-'}
     testLogger:plot()
  end
  --confusion:zero()

  --for i=1,(table.getn(names))do
  --      print(names[i])
  --end
  -- Export:
  print(positionSize)      
end
confusion:zero()
end