import numpy as np
import vtk
from PyQt5.QtCore import QFile


def read_file(file_path):
    file = QFile(file_path)
    if not file.open(QFile.ReadOnly):
        print(f"Failed to open file: {file_path}")
        return None

    data = file.readAll()
    file.close()

    return bytes(data)


data1 = read_file('TT/CT.rdata')
data2 = read_file('TT/MR_TO_CT.rdata')

with open('TT/CT.rdata.header', 'r') as f:
    lines = f.readlines()

sizes_index = next(i for i, line in enumerate(lines) if line.startswith('SIZES:'))

dimensions = list(map(int, lines[sizes_index + 1].split()))

expected_size = np.prod(dimensions)

array1 = np.copy(np.frombuffer(data1, dtype=np.int16).reshape(dimensions))
array2 = np.copy(np.frombuffer(data2, dtype=np.int16).reshape(dimensions))

voxel_dims_index = next(i for i, line in enumerate(lines) if line.startswith('VOXEL_DIMS:'))
voxel_dims = list(map(float, lines[voxel_dims_index + 1].split()))

vtk_data1 = vtk.vtkImageData()
vtk_data1.SetDimensions(array1.shape)

vtk_data2 = vtk.vtkImageData()
vtk_data2.SetDimensions(array2.shape)

flat_array1 = np.ravel(array1).astype(np.int16)

vtk_array1 = vtk.vtkShortArray()

vtk_array1.SetArray(flat_array1, flat_array1.size, 1)

vtk_data1.GetPointData().SetScalars(vtk_array1)

flat_array2 = np.ravel(array2).astype(np.int16)

vtk_array2 = vtk.vtkShortArray()

vtk_array2.SetArray(flat_array2, flat_array2.size, 1)

vtk_data2.GetPointData().SetScalars(vtk_array2)

vtk_data2.GetPointData().SetScalars(vtk_array2)

vtk_data1.SetSpacing(voxel_dims)
vtk_data2.SetSpacing(voxel_dims)

volume1 = vtk.vtkVolume()
volume2 = vtk.vtkVolume()
mapper1 = vtk.vtkSmartVolumeMapper()
mapper1.SetInputData(vtk_data1)

volume1.SetMapper(mapper1)

mapper2 = vtk.vtkSmartVolumeMapper()
mapper2.SetInputData(vtk_data2)

volume2.SetMapper(mapper2)

colorFunc2 = vtk.vtkColorTransferFunction()
colorFunc2.AddRGBPoint(array2.min(), 0.0, 0.0, 0.0)
colorFunc2.AddRGBPoint(array2.max() / 3, 0.6, 0.6, 0.6)
colorFunc2.AddRGBPoint(array2.max() * 2 / 3, 0.9, 0.9, 0.9)
colorFunc2.AddRGBPoint(array2.max(), 1.0, 1.0, 1.0)

volume_scalar_opacity2 = vtk.vtkPiecewiseFunction()
volume_scalar_opacity2.AddPoint(0, 0.00)
volume_scalar_opacity2.AddPoint(array2.max() / 4, 0.15)
volume_scalar_opacity2.AddPoint(array2.max() / 2, 0.15)
volume_scalar_opacity2.AddPoint(array2.max(), 0.85)

volumeProperty2 = vtk.vtkVolumeProperty()
volumeProperty2.SetColor(colorFunc2)
volumeProperty2.SetScalarOpacity(volume_scalar_opacity2)

colorFunc1 = vtk.vtkColorTransferFunction()
colorFunc1.AddRGBPoint(array1.min(), 0.0, 0.0, 0.0)
colorFunc1.AddRGBPoint(array1.max() / 3, 0.6, 0.6, 0.6)
colorFunc1.AddRGBPoint(array1.max() * 2 / 3, 0.9, 0.9, 0.9)
colorFunc1.AddRGBPoint(array1.max(), 1.0, 1.0, 1.0)

volume_scalar_opacity1 = vtk.vtkPiecewiseFunction()
volume_scalar_opacity1.AddPoint(0, 0.00)
volume_scalar_opacity1.AddPoint(2900, 0.00)
volume_scalar_opacity1.AddPoint(array1.max(), 1)

volumeProperty1 = vtk.vtkVolumeProperty()
volumeProperty1.SetColor(colorFunc1)
volumeProperty1.SetScalarOpacity(volume_scalar_opacity1)

# Get the bounds of the volume
bounds1 = vtk_data1.GetBounds()
bounds2 = vtk_data2.GetBounds()

# Calculate the center of the volumes
center_x = (bounds1[0] + bounds1[1] + bounds2[0] + bounds2[1]) / 4
center_y = (bounds1[2] + bounds1[3] + bounds2[2] + bounds2[3]) / 4
center_z = (bounds1[4] + bounds1[5] + bounds2[4] + bounds2[5]) / 4

plane1 = vtk.vtkPlane()
plane1.SetOrigin(center_x, center_y, center_z)
plane1.SetNormal(1, 0, 0)
plane2 = vtk.vtkPlane()
plane2.SetOrigin(center_x, center_y, center_z)
plane2.SetNormal(-1, 0, 0)

mapper1.AddClippingPlane(plane1)
mapper2.AddClippingPlane(plane2)

cutter1 = vtk.vtkCutter()
cutter1.SetCutFunction(plane1)
cutter1.SetInputData(volume1.GetMapper().GetInput())
cutter1.Update()

cutter2 = vtk.vtkCutter()
cutter2.SetCutFunction(plane2)
cutter2.SetInputData(volume2.GetMapper().GetInput())
cutter2.Update()

cutterMapper1 = vtk.vtkPolyDataMapper()
cutterMapper1.SetInputConnection(cutter1.GetOutputPort())
cutterMapper2 = vtk.vtkPolyDataMapper()
cutterMapper2.SetInputConnection(cutter2.GetOutputPort())

planeActor1 = vtk.vtkActor()
planeActor1.GetProperty().SetLineWidth(2)
planeActor1.SetMapper(cutterMapper1)
planeActor1.GetProperty().SetColor(1, 1, 1)

planeActor2 = vtk.vtkActor()
planeActor2.GetProperty().SetLineWidth(2)
planeActor2.SetMapper(cutterMapper2)
planeActor2.GetProperty().SetColor(1, 1, 1)

volume1.SetProperty(volumeProperty1)
volume2.SetProperty(volumeProperty2)

histogram = vtk.vtkImageHistogram()
histogram.SetInputData(vtk_data1)
histogram.GenerateHistogramImageOn()
histogram.SetHistogramImageSize(1000, 1000)
histogram.SetHistogramImageScaleToLog()
histogram.AutomaticBinningOn()
histogram.Update()
histogram.GetOutputPort()
histogram_data = histogram.GetHistogram()
max = histogram_data.GetTuple1(1)
max_index = 0
for i in range(1, histogram_data.GetNumberOfTuples()):
    if histogram_data.GetTuple1(i) > max:
        max = histogram_data.GetTuple1(i)
        max_index = i

logo = vtk.vtkLogoRepresentation()
logo.SetImage(histogram.GetOutput())

widget = vtk.vtkLogoWidget()
widget.SetRepresentation(logo)

renderer = vtk.vtkRenderer()

renderer.AddVolume(volume1)
renderer.AddVolume(volume2)
renderer.AddActor(planeActor1)
renderer.AddActor(planeActor2)
renderer.SetBackground(0.2, 0.4, 0.6)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

screen_width, screen_height = render_window.GetScreenSize()
render_window.SetSize(screen_width, screen_height)


def key_press_event(obj, event):
    key = obj.GetKeySym()
    if key == "Left":
        plane1.SetOrigin(plane1.GetOrigin()[0] - 1, plane1.GetOrigin()[1], plane1.GetOrigin()[2])
        plane2.SetOrigin(plane2.GetOrigin()[0] - 1, plane2.GetOrigin()[1], plane2.GetOrigin()[2])
    elif key == "Right":
        plane1.SetOrigin(plane1.GetOrigin()[0] + 1, plane1.GetOrigin()[1], plane1.GetOrigin()[2])
        plane2.SetOrigin(plane2.GetOrigin()[0] + 1, plane2.GetOrigin()[1], plane2.GetOrigin()[2])
    cutter1.Update()
    cutter2.Update()
    render_window.Render()


interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
widget.SetInteractor(interactor)
widget.On()
interactor.AddObserver("KeyPressEvent", key_press_event)
interactor.Initialize()
interactor.Start()
