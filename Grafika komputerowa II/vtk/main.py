import os

import vtk
from PIL import Image
import numpy as np

image = Image.open('heightmap.png').convert('RGB')
image_array = np.array(image)

height, width, _ = image_array.shape

max_pixel_value = 189
min_pixel_value = 68

skybox_texture = vtk.vtkTexture()
skybox_texture.CubeMapOn()
i = 0
reader = vtk.vtkImageReader2Factory()
for file in os.listdir("skybox"):
    img_reader = reader.CreateImageReader2(f"skybox/{file}")
    img_reader.SetFileName(f"skybox/{file}")
    flip = vtk.vtkImageFlip()
    flip.SetInputConnection(img_reader.GetOutputPort(0))
    flip.SetFilteredAxis(1)
    skybox_texture.SetInputConnection(i, flip.GetOutputPort())
    i += 1

skybox = vtk.vtkSkybox()
skybox.SetProjectionToCube()
skybox.SetTexture(skybox_texture)

logo = vtk.vtkLogoRepresentation()
img_reader = reader.CreateImageReader2('widget.png')
img_reader.SetFileName('widget.png')
img_reader.Update()
logo.SetImage(img_reader.GetOutput())

widget = vtk.vtkLogoWidget()
widget.SetRepresentation(logo)

image_data = vtk.vtkImageData()
image_data.SetDimensions(width, height, 256)
image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

for y in range(height):
    for x in range(width):
        r, g, b = map(int, image_array[y, x])

        pixel_value = (r + g + b) // 3
        scalar_value = int(100 * (pixel_value - min_pixel_value) / (max_pixel_value - min_pixel_value))
        for z in range(scalar_value):
            image_data.SetScalarComponentFromFloat(x, y, z, 0, r)
            image_data.SetScalarComponentFromFloat(x, y, z, 1, g)
            image_data.SetScalarComponentFromFloat(x, y, z, 2, b)

luminance = vtk.vtkImageLuminance()
luminance.SetInputData(image_data)

append = vtk.vtkImageAppendComponents()
append.AddInputData(image_data)
append.AddInputConnection(luminance.GetOutputPort())

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.ShadeOn()
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.SetIndependentComponents(False)

colorFunc = vtk.vtkColorTransferFunction()
colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
colorFunc.AddRGBPoint(255, 1.0, 1.0, 1.0)

opacityFunc = vtk.vtkPiecewiseFunction()
opacityFunc.AddPoint(0, 0.0)
opacityFunc.AddPoint(255, 1.0)

volumeProperty.SetColor(colorFunc)
volumeProperty.SetScalarOpacity(opacityFunc)

volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
volumeMapper.SetInputConnection(append.GetOutputPort())

volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

renderer.AddVolume(volume)
renderer.SetBackground(1.0, 1.0, 1.0)
renderer.AddActor(skybox)
widget.SetInteractor(renderWindowInteractor)
widget.On()

renderWindow.SetSize(600, 600)

camera = renderer.GetActiveCamera()

bounds = volume.GetBounds()

camera.SetPosition(bounds[0], (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2)

camera.SetFocalPoint((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2)
camera.Roll(90)
renderWindow.Render()
renderWindowInteractor.Start()
