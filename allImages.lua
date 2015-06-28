-------------------------------------------------
-- Author : Charan
--This script is used for person detection,
-- Draws a rectangle around the detected object
--Uses a previously generated model
-------------------------------------------------


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


--loop through all images

opt = lapp[[
   -n, --network  (default 'model.net')   path to networkimage.
   -s, --save      (default '')
   -t, --threads  (default 8)             number of threads
   -b, --batchSize (default 128)
]]
function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!
--dofile 'runonScaledImages.lua'
dofile 'runonScaledImagesResize.lua'  --removed manually unsizing
--local testDir = 'pedestrian/'
local testDir = 'INRIAPerson/70X134H96/Test/pos/'
local testImaNumber = #ls(testDir)
for i=1,testImaNumber-1 do
	runOnScaledImages(opt,i)
end



