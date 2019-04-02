#Jednoduchy generator rontgenovych snimkov z CT
#Autor: Michaela Dronzekova, xdronz00

import vtk
import msvcrt
import os
import sys

#pouzitie: python XRayGenerator.py D:\Downloads\SMIR_191_20190218_141259\SMIR.Head.060Y.F.CT.191 D:\results


zoom = 1
camerax = -1
cameray = 2
cameraz = -0.5
mode = False
color_level = 0.5
window_level = 1.0
print(len(sys.argv))
if len(sys.argv) < 4:
    counter = 0
else:
    counter = int(sys.argv[3])
save_directory = sys.argv[2]
def keyPressed(obj,event):
    global renderer
    global zoom
    global renderWindow
    global interactionRenderer
    global mode
    global counter
    global rayCastMapper
    global window_level
    global color_level
    key = obj.GetKeySym()
    # z == zoom in
    # o == zoom out
    # up, down, left, right == move camera or rotate
    # s == shift mode from move to rotate and back
    # r == render, for now only into one folder
    if key == "Up" or key == "Down" or key == "Left" or key == "Right" or key =="g":
        lastXYpos = interactionRenderer.GetLastEventPosition()
        lastX = lastXYpos[0]
        lastY = lastXYpos[1]
        center = renderWindow.GetSize()
        centerX = center[0]/2.0
        centerY = center[1]/2.0
    if key == "z":
       if zoom < 1:
            zoom = 1
       zoom = 1.05
       renderer.GetActiveCamera().Zoom(zoom)

    elif key == "o":
        if zoom > 1:
            zoom = 1
        zoom = 0.95
        renderer.GetActiveCamera().Zoom(zoom)
    
    elif key == "Up":
        
        if mode == False:
            y = lastY+10
            Pan(renderer,renderer.GetActiveCamera(),lastX,y,lastX,lastY,centerX,centerY )
        else:
            y = lastY + 5
            Rotate(renderer,renderer.GetActiveCamera(),lastX,y,lastX,lastY,centerX,centerY)
    elif key == "Down":
        
        if mode == False:
            y = lastY - 10
            Pan(renderer,renderer.GetActiveCamera(),lastX,y,lastX,lastY,centerX,centerY )
        else:
            y = lastY - 5
            Rotate(renderer,renderer.GetActiveCamera(),lastX,y,lastX,lastY,centerX,centerY)
    elif key == "Left":
        
        if mode == False:
            x = lastX -10
            Pan(renderer,renderer.GetActiveCamera(),x,lastY,lastX,lastY,centerX,centerY )
        else:
            x = lastX -5
            Rotate(renderer,renderer.GetActiveCamera(),x,lastY,lastX,lastY,centerX,centerY)
    elif key == "Right":
        
        if mode == False:
            x = lastX + 10
            Pan(renderer,renderer.GetActiveCamera(),x,lastY,lastX,lastY,centerX,centerY )
        else:
            x = lastX + 5
            Rotate(renderer,renderer.GetActiveCamera(),x,lastY,lastX,lastY,centerX,centerY)
    elif key == "s":
        mode = not mode

    #vygenerovanie snimku hlavy
    elif key == "h":
        string = "head"
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renderWindow)
        windowToImageFilter.Update()
        writer = vtk.vtkPNGWriter()
        if not os.path.exists(save_directory+"/"+string):
            os.makedirs(save_directory+"/"+string)
        writer.SetFileName(save_directory+"/"+string+"/"+string+str(counter)+".png")
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        counter += 1
        writer.Write()

    #vygenerovanie snimku panvy
    elif key == "t":
        string = "pelvis"
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renderWindow)
        windowToImageFilter.Update()
        writer = vtk.vtkPNGWriter()
        if not os.path.exists(save_directory+"/"+string):
            os.makedirs(save_directory+"/"+string)
        writer.SetFileName(save_directory+"/"+string+"/"+string+str(counter)+".png")
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        counter += 1
        writer.Write()

    #vygenerovanie snimku hornych a dolnych koncatin
    elif key == "l":
        string = "long_bones"
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renderWindow)
        windowToImageFilter.Update()
        writer = vtk.vtkPNGWriter()
        if not os.path.exists(save_directory+"/"+string):
            os.makedirs(save_directory+"/"+string)
        writer.SetFileName(save_directory+"/"+string+"/"+string+str(counter)+".png")
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        counter += 1
        writer.Write()

    elif key == "g":
        counter = 0
        while counter<46:
            # Rotate(renderer,renderer.GetActiveCamera(),x,lastY,lastX,lastY,centerX,centerY)
            if counter == 0:
                camera.Azimuth(0)
            elif counter <= 10:
                camera.Azimuth(1)
            elif counter <= 15:
                camera.Azimuth(2)
            elif counter <= 31:
                camera.Azimuth(20)
            elif counter <= 36:
                camera.Azimuth(2)
            elif counter <= 45:
                camera.Azimuth(1)

            renderWindow.Render()

            if counter <= 5:
                string = str(0)
            elif counter <= 10:
                string = str(1)
            elif counter <= 15:
                string = str(2)
            elif counter <= 30:
                string = str(3)
            elif counter <= 35:
                string = str(4)
            elif counter <= 40:
                string = str(5)
            elif counter <= 45:
                string = str(6)
            
            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(renderWindow)
            windowToImageFilter.Update()
            writer = vtk.vtkPNGWriter()
            if not os.path.exists(save_directory+"/"+string):
                os.makedirs(save_directory+"/"+string)
            writer.SetFileName(save_directory+"/"+string+"/"+filename + str(counter) +".png")
            writer.SetInputConnection(windowToImageFilter.GetOutputPort())
            counter += 1
            writer.Write()

    #znizenie jasu
    elif key == "1":
        color_level+=0.05
        rayCastMapper.SetFinalColorLevel(color_level)
    #zvysenie jasu
    elif key == "2":
        color_level-=0.05
        rayCastMapper.SetFinalColorLevel(color_level)
    #znizenie kontrasu
    elif key == "3":
        window_level+=0.05
        rayCastMapper.SetFinalColorWindow(window_level)
    #zvysenie kontrastu
    elif key == "4":
        window_level-=0.05
        rayCastMapper.SetFinalColorWindow(window_level)
        
    if key != "t" and key != "h" and key != "l":
        renderWindow.Render()
       
#posuvanie modelu
def Pan(renderer,camera,x,y, lastX, lastY, centerX, centerY):
    FPoint = camera.GetFocalPoint()
    FPoint0 = FPoint[0]
    FPoint1 = FPoint[1]
    FPoint2 = FPoint[2]

    PPoint = camera.GetPosition()
    PPoint0 = PPoint[0]
    PPoint1 = PPoint[1]
    PPoint2 = PPoint[2]
    renderer.SetWorldPoint(FPoint0, FPoint1, FPoint2, 1.0)
    renderer.WorldToDisplay()
    DPoint = renderer.GetDisplayPoint()
    focalDepth = DPoint[2]

    APoint0 = centerX+(x-lastX)
    APoint1 = centerY+(y-lastY)

    renderer.SetDisplayPoint(APoint0, APoint1, focalDepth)
    renderer.DisplayToWorld()
    RPoint = renderer.GetWorldPoint()
    RPoint0 = RPoint[0]
    RPoint1 = RPoint[1]
    RPoint2 = RPoint[2]
    RPoint3 = RPoint[3]

    if RPoint3 != 0.0:
        RPoint0 = RPoint0/RPoint3
        RPoint1 = RPoint1/RPoint3
        RPoint2 = RPoint2/RPoint3

    camera.SetFocalPoint( (FPoint0-RPoint0)/2.0 + FPoint0,
                          (FPoint1-RPoint1)/2.0 + FPoint1,
                          (FPoint2-RPoint2)/2.0 + FPoint2)
    camera.SetPosition( (FPoint0-RPoint0)/2.0 + PPoint0,
                        (FPoint1-RPoint1)/2.0 + PPoint1,
                        (FPoint2-RPoint2)/2.0 + PPoint2)
    #renWin.Render()

#Rotacia modelu
def Rotate(renderer, camera, x, y, lastX, lastY, centerX, centerY):
    camera.Azimuth(lastX-x)
    camera.Elevation(lastY-y)
    camera.OrthogonalizeViewUp()
    print(x)
    print(y)
    print("rotace")

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)

#vykreslenie
print("vykresleni")
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.0,0.0,0.0)
camera = renderer.GetActiveCamera()
camera.ParallelProjectionOff()
camera.SetViewUp(0,0,-1)
camera.SetPosition(-1,2,-0.5)
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(640,480)
renderWindow.AddRenderer(renderer)
interactionRenderer = vtk.vtkRenderWindowInteractor()
interactionRenderer.SetRenderWindow(renderWindow)
interactionRenderer.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
interactionRenderer.AddObserver("KeyPressEvent",keyPressed)

#nacitanie obrazkov
print("nacteni")
PathDicom=str(sys.argv[1])
print(PathDicom)
filename = os.path.basename(PathDicom)
print(filename)
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(PathDicom)
reader.Update()
pom = reader.GetOutput().GetScalarRange()

print("vykresleni1")
t = vtk.vtkImageShiftScale()
t.SetInputConnection(reader.GetOutputPort())
t.SetShift(-pom[0])
magnitude = pom[1] - pom[0]
if magnitude < 0.00000000001:
    magnitude=1.0

print("vykresleni2")
t.SetScale(255.0/magnitude)
t.SetOutputScalarTypeToUnsignedChar()
t.Update()

print("vykresleni3")
colorTransferFunction = vtk.vtkColorTransferFunction()
colorTransferFunction.AddRGBPoint(   0.0, 0.0,0.0,0.0)
colorTransferFunction.AddRGBPoint( 500.0, 0.9,0.5,0.3)
colorTransferFunction.AddRGBPoint(1100.0, 0.8,0.8,0.6)
colorTransferFunction.AddRGBPoint(1200.0, 0.6,0.6,0.6)

print("vykresleni4")
opacityTransferFunction = vtk.vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(    0, 0.0);
opacityTransferFunction.AddPoint(  1200, 0.05)
opacityTransferFunction.AddPoint( 1300, 1.0)

print("vykresleni5")
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetScalarOpacity(opacityTransferFunction)
rayCastMapper = vtk.vtkGPUVolumeRayCastMapper()
rayCastMapper.SetInputData( t.GetOutput())
rayCastMapper.SetBlendModeToAdditive()
rayCastMapper.SetFinalColorLevel(0.5)
rayCastMapper.SetFinalColorWindow(1.0)


print("vykresleni6")
volData = vtk.vtkVolume()
volData.SetMapper(rayCastMapper)
volData.SetProperty(volumeProperty)
renderer.AddVolume(volData)
renderer.ResetCamera()
renderer.GetActiveCamera().Dolly(1)
renderer.ResetCameraClippingRange()

print("vykresleni7")
interactionRenderer.Initialize()
interactionRenderer.Start()
print('end')


