# -*- coding: mbcs -*-
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import numpy as np
import csv
import os
from os import path

session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE) 
Mdb()

ShearMod_Sil_950 = 0.8
Jm_Sil_950 = 6.0

# define parameters
Meshsize = 3.0
# P = 0.09
execfile("Parameters_P.py")
execfile("Parameters_for_Fpn_bending_Gravity.py")

# import part
mdb.openStep('F:/Fast-pn-actuator/Fpn_bending/Board/check/assembleBody.STEP',
    scaleFromFile=ON)
mdb.models['Model-1'].PartFromGeometryFile(combine=False, dimensionality=
    THREE_D, geometryFile=mdb.acis, name='Abaqus_bottomA', type=
    DEFORMABLE_BODY)
mdb.models['Model-1'].PartFromGeometryFile(bodyNum=3, combine=False, 
    dimensionality=THREE_D, geometryFile=mdb.acis, name='Abaqus_bottomB', type=
    DEFORMABLE_BODY)
mdb.models['Model-1'].PartFromGeometryFile(bodyNum=2, combine=False, 
    dimensionality=THREE_D, geometryFile=mdb.acis, name='Abaqus_main', type=
    DEFORMABLE_BODY)

# # yeoh model material
# mdb.models['Model-1'].Material(name='Elastosil')
# mdb.models['Model-1'].materials['Elastosil'].Hyperelastic(materialType=
#     ISOTROPIC, table=((0.22766, 0.05897, -0.00451676, 0.044, 0.0, 0.0), ), 
#     testData=OFF, type=YEOH, volumetricResponse=VOLUMETRIC_DATA)
# mdb.models['Model-1'].materials['Elastosil'].Density(table=((1e-09, ), ))
# mdb.models['Model-1'].materials['Elastosil'].Viscoelastic(domain=TIME, table=((
#     0.05253, 0.0, 5.93164), (0.04793, 0.0, 47.6395), (0.0511, 0.0, 320.823), (
#     0.0996, 0.0, 2961.89)), time=PRONY)
# Gent Model material
mdb.models['Model-1'].Material(name='Elastosil')
mdb.models['Model-1'].materials['Elastosil'].Depvar(n=2)
mdb.models['Model-1'].materials['Elastosil'].Hyperelastic(materialType=ISOTROPIC, 
    testData=OFF, type=USER, properties=2, table=((ShearMod_Sil_950, Jm_Sil_950), ))
mdb.models['Model-1'].materials['Elastosil'].Density(table=((1.0e-09, ), ))
# Paper material
mdb.models['Model-1'].Material(name='Paper')
mdb.models['Model-1'].materials['Paper'].Density(table=((7.5e-10, ), ))
mdb.models['Model-1'].materials['Paper'].Elastic(table=((6500.0, 0.2), ))


# section
mdb.models['Model-1'].HomogeneousSolidSection(material='Elastosil', name=
    'Sec-Elastosil', thickness=None)
mdb.models['Model-1'].HomogeneousShellSection(idealization=NO_IDEALIZATION, 
    integrationRule=SIMPSON, material='Paper', name='Sec-Paper', 
    nodalThicknessField='', numIntPts=5, poissonDefinition=DEFAULT, 
    preIntegrate=OFF, temperature=GRADIENT, thickness=0.1, thicknessField='', 
    thicknessModulus=None, thicknessType=UNIFORM, useDensity=OFF)

# section-assignment
mdb.models['Model-1'].parts['Abaqus_bottomA'].SectionAssignment(offset=0.0, 
    offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
    cells=mdb.models['Model-1'].parts['Abaqus_bottomA'].cells),
    sectionName='Sec-Elastosil', thicknessAssignment=FROM_SECTION)
mdb.models['Model-1'].parts['Abaqus_bottomB'].SectionAssignment(offset=0.0, 
    offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
    cells=mdb.models['Model-1'].parts['Abaqus_bottomB'].cells),
    sectionName='Sec-Elastosil', thicknessAssignment=FROM_SECTION)
mdb.models['Model-1'].parts['Abaqus_main'].SectionAssignment(offset=0.0, 
    offsetField='', offsetType=MIDDLE_SURFACE, region=Region(
    cells=mdb.models['Model-1'].parts['Abaqus_main'].cells),
    sectionName='Sec-Elastosil', thicknessAssignment=FROM_SECTION)

# define partB surface
mdb.models['Model-1'].parts['Abaqus_bottomB'].Surface(name='Top of B', 
    side1Faces=mdb.models['Model-1'].parts['Abaqus_bottomB'].faces.findAt(((
    10.73805, -5.036239, 120.333333), )))

# Assembly and merge
mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name=
    'Abaqus_bottomA-1', part=mdb.models['Model-1'].parts['Abaqus_bottomA'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name=
    'Abaqus_bottomB-1', part=mdb.models['Model-1'].parts['Abaqus_bottomB'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Abaqus_main-1', 
    part=mdb.models['Model-1'].parts['Abaqus_main'])
mdb.models['Model-1'].rootAssembly.InstanceFromBooleanMerge(domain=GEOMETRY, 
    instances=(
    mdb.models['Model-1'].rootAssembly.instances['Abaqus_bottomA-1'], 
    mdb.models['Model-1'].rootAssembly.instances['Abaqus_bottomB-1'], 
    mdb.models['Model-1'].rootAssembly.instances['Abaqus_main-1']), 
    keepIntersections=ON, name='Merged', originalInstances=SUPPRESS)

# define shell skin
mdb.models['Model-1'].parts['Merged'].Skin(faces=
    mdb.models['Model-1'].parts['Merged'].faces.findAt(((10.73805, -5.036239, 120.333333), 
    )), name='Skin-1')
mdb.models['Model-1'].parts['Merged'].SectionAssignment(offset=0.0, 
    offsetField='', offsetType=MIDDLE_SURFACE, region=Region(skinFaces=((
    'Skin-1', mdb.models['Model-1'].parts['Merged'].faces.findAt(((
    10.73805, -5.036239, 120.333333), ))), )), sectionName='Sec-Paper',
    thicknessAssignment=FROM_SECTION)

# define cavity
mdb.models['Model-1'].rootAssembly.regenerate()
mdb.models['Model-1'].parts['Merged'].Surface(name='Surf-Inner Cavity', 
    side1Faces=mdb.models['Model-1'].parts['Merged'].faces.findAt(((
    36.071384, 9.963761, 117.333333), ), ((27.071383, 9.963761, 117.333333), ), ((18.071383, 
    9.963761, 117.333333), ), ((9.071383, 9.963761, 117.333333), ), ((0.071383, 
    9.963761, 117.333333), ), ((-8.928617, 9.963761, 117.333333), ), ((
    -17.928617, 9.963761, 117.333333), ), ((-26.928617, 9.963761, 117.333333), 
    ), ((-35.928616, 9.963761, 117.333333), ), ((-44.928616, 9.963761, 
    117.333333), ), ((-53.928616, 9.963761, 117.333333), ), ((-47.928616, 
    -1.036239, 114.666667), ), ((-38.928616, -1.036239, 114.666667), ), ((
    -29.928617, -1.036239, 114.666667), ), ((-20.928617, -1.036239, 
    114.666667), ), ((-11.928617, -1.036239, 114.666667), ), ((-2.928617, 
    -1.036239, 114.666667), ), ((6.071383, -1.036239, 114.666667), ), ((
    15.071383, -1.036239, 114.666667), ), ((24.071383, -1.036239, 114.666667), 
    ), ((33.071384, -1.036239, 114.666667), ), ((36.071384, 5.36376, 108.0), ), 
    ((38.071383, 0.763761, 112.666667), ), ((37.071384, 0.763761, 122.0), ), ((
    27.071383, 5.36376, 108.0), ), ((33.071384, -2.902906, 116.0), ), ((
    31.071383, -1.969573, 114.0), ), ((29.071383, 1.697094, 110.0), ), ((
    28.071383, 0.763761, 122.0), ), ((18.071383, 5.36376, 108.0), ), ((
    24.071383, -2.902906, 116.0), ), ((22.071383, -1.969573, 114.0), ), ((
    20.071383, 1.697094, 110.0), ), ((19.071383, 0.763761, 122.0), ), ((
    9.071383, 5.36376, 108.0), ), ((15.071383, -2.902906, 116.0), ), ((
    13.071383, -1.969573, 114.0), ), ((11.071383, 1.697094, 110.0), ), ((
    10.071383, 0.763761, 122.0), ), ((0.071383, 5.36376, 108.0), ), ((6.071383, 
    -2.902906, 116.0), ), ((4.071383, -1.969573, 114.0), ), ((2.071383, 
    1.697094, 110.0), ), ((1.071383, 0.763761, 122.0), ), ((-8.928617, 5.36376, 
    108.0), ), ((-2.928617, -2.902906, 116.0), ), ((-4.928617, -1.969573, 
    114.0), ), ((-6.928617, 1.697094, 110.0), ), ((-7.928617, 0.763761, 122.0), 
    ), ((-17.928617, 5.36376, 108.0), ), ((-11.928617, -2.902906, 116.0), ), ((
    -13.928617, -1.969573, 114.0), ), ((-15.928617, 1.697094, 110.0), ), ((
    -16.928617, 0.763761, 122.0), ), ((-26.928617, 5.36376, 108.0), ), ((
    -20.928617, -2.902906, 116.0), ), ((-22.928617, -1.969573, 114.0), ), ((
    -24.928617, 1.697094, 110.0), ), ((-25.928617, 0.763761, 122.0), ), ((
    -35.928616, 5.36376, 108.0), ), ((-29.928617, -2.902906, 116.0), ), ((
    -31.928616, -1.969573, 114.0), ), ((-33.928617, 1.697094, 110.0), ), ((
    -34.928616, 0.763761, 122.0), ), ((-44.928616, 5.36376, 108.0), ), ((
    -38.928616, -2.902906, 116.0), ), ((-40.928616, -1.969573, 114.0), ), ((
    -42.928617, 1.697094, 110.0), ), ((-43.928616, 0.763761, 122.0), ), ((
    -53.928616, 5.36376, 108.0), ), ((-47.928616, -2.902906, 116.0), ), ((
    -49.928616, -1.969573, 114.0), ), ((-51.928617, 1.697094, 110.0), ), ((
    -52.928616, 0.763761, 122.0), ), ((-54.928617, 0.763761, 117.333333), ), ((
    -45.928617, 1.697094, 110.0), ), ((-36.928617, 1.697094, 110.0), ), ((
    -27.928617, 1.697094, 110.0), ), ((-18.928617, 1.697094, 110.0), ), ((
    -9.928617, 1.697094, 110.0), ), ((-0.928617, 1.697094, 110.0), ), ((
    8.071383, 1.697094, 110.0), ), ((17.071383, 1.697094, 110.0), ), ((
    26.071383, 1.697094, 110.0), ), ((35.071383, 1.697094, 110.0), ), ((
    -52.928616, -3.836239, 117.333333), ) ))
mdb.models['Model-1'].parts['Merged'].Set(faces=
    mdb.models['Model-1'].parts['Merged'].faces.findAt(((36.071384, 9.963761, 
    117.333333), ), ((27.071383, 9.963761, 117.333333), ), ((18.071383, 
    9.963761, 117.333333), ), ((9.071383, 9.963761, 117.333333), ), ((0.071383, 
    9.963761, 117.333333), ), ((-8.928617, 9.963761, 117.333333), ), ((
    -17.928617, 9.963761, 117.333333), ), ((-26.928617, 9.963761, 117.333333), 
    ), ((-35.928616, 9.963761, 117.333333), ), ((-44.928616, 9.963761, 
    117.333333), ), ((-53.928616, 9.963761, 117.333333), ), ((-47.928616, 
    -1.036239, 114.666667), ), ((-38.928616, -1.036239, 114.666667), ), ((
    -29.928617, -1.036239, 114.666667), ), ((-20.928617, -1.036239, 
    114.666667), ), ((-11.928617, -1.036239, 114.666667), ), ((-2.928617, 
    -1.036239, 114.666667), ), ((6.071383, -1.036239, 114.666667), ), ((
    15.071383, -1.036239, 114.666667), ), ((24.071383, -1.036239, 114.666667), 
    ), ((33.071384, -1.036239, 114.666667), ), ((36.071384, 5.36376, 108.0), ), 
    ((38.071383, 0.763761, 112.666667), ), ((37.071384, 0.763761, 122.0), ), ((
    27.071383, 5.36376, 108.0), ), ((33.071384, -2.902906, 116.0), ), ((
    31.071383, -1.969573, 114.0), ), ((29.071383, 1.697094, 110.0), ), ((
    28.071383, 0.763761, 122.0), ), ((18.071383, 5.36376, 108.0), ), ((
    24.071383, -2.902906, 116.0), ), ((22.071383, -1.969573, 114.0), ), ((
    20.071383, 1.697094, 110.0), ), ((19.071383, 0.763761, 122.0), ), ((
    9.071383, 5.36376, 108.0), ), ((15.071383, -2.902906, 116.0), ), ((
    13.071383, -1.969573, 114.0), ), ((11.071383, 1.697094, 110.0), ), ((
    10.071383, 0.763761, 122.0), ), ((0.071383, 5.36376, 108.0), ), ((6.071383, 
    -2.902906, 116.0), ), ((4.071383, -1.969573, 114.0), ), ((2.071383, 
    1.697094, 110.0), ), ((1.071383, 0.763761, 122.0), ), ((-8.928617, 5.36376, 
    108.0), ), ((-2.928617, -2.902906, 116.0), ), ((-4.928617, -1.969573, 
    114.0), ), ((-6.928617, 1.697094, 110.0), ), ((-7.928617, 0.763761, 122.0), 
    ), ((-17.928617, 5.36376, 108.0), ), ((-11.928617, -2.902906, 116.0), ), ((
    -13.928617, -1.969573, 114.0), ), ((-15.928617, 1.697094, 110.0), ), ((
    -16.928617, 0.763761, 122.0), ), ((-26.928617, 5.36376, 108.0), ), ((
    -20.928617, -2.902906, 116.0), ), ((-22.928617, -1.969573, 114.0), ), ((
    -24.928617, 1.697094, 110.0), ), ((-25.928617, 0.763761, 122.0), ), ((
    -35.928616, 5.36376, 108.0), ), ((-29.928617, -2.902906, 116.0), ), ((
    -31.928616, -1.969573, 114.0), ), ((-33.928617, 1.697094, 110.0), ), ((
    -34.928616, 0.763761, 122.0), ), ((-44.928616, 5.36376, 108.0), ), ((
    -38.928616, -2.902906, 116.0), ), ((-40.928616, -1.969573, 114.0), ), ((
    -42.928617, 1.697094, 110.0), ), ((-43.928616, 0.763761, 122.0), ), ((
    -53.928616, 5.36376, 108.0), ), ((-47.928616, -2.902906, 116.0), ), ((
    -49.928616, -1.969573, 114.0), ), ((-51.928617, 1.697094, 110.0), ), ((
    -52.928616, 0.763761, 122.0), ), ((-54.928617, 0.763761, 117.333333), ), ((
    -45.928617, 1.697094, 110.0), ), ((-36.928617, 1.697094, 110.0), ), ((
    -27.928617, 1.697094, 110.0), ), ((-18.928617, 1.697094, 110.0), ), ((
    -9.928617, 1.697094, 110.0), ), ((-0.928617, 1.697094, 110.0), ), ((
    8.071383, 1.697094, 110.0), ), ((17.071383, 1.697094, 110.0), ), ((
    26.071383, 1.697094, 110.0), ), ((35.071383, 1.697094, 110.0), ), ((
    -52.928616, -3.836239, 117.333333), ), ),
    name='Set-Inner Cavity')
RP_1 = mdb.models['Model-1'].rootAssembly.ReferencePoint(point=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].InterestingPoint(
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].edges.findAt((
    -9.928617, -3.836239, 122.0), ), MIDDLE))
mdb.models['Model-1'].rootAssembly.Set(name='Set-RP', referencePoints=(
    mdb.models['Model-1'].rootAssembly.referencePoints[RP_1.id], ))

# define BC
mdb.models['Model-1'].EncastreBC(createStepName='Initial', localCsys=None, name=
    'BC-1', region=Region(
    faces=mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(
    ((-59.928617, 1.963761, 111.666667), ), )))

# define step load for gravity
mdb.models['Model-1'].rootAssembly.Set(cells=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].cells,name='Set-6')
mdb.models['Model-1'].StaticStep(initialInc=1e-02, maxNumInc=1000, minInc=
    1e-4, maxInc=1, name='Step-Gravity', nlgeom=ON, previous='Initial')
mdb.models['Model-1'].Gravity(comp2=-9810.0, createStepName='Step-Gravity', 
    distributionType=UNIFORM, field='', name='Load-1', region=
    mdb.models['Model-1'].rootAssembly.sets['Set-6'])

# define step load for pressure
mdb.models['Model-1'].StaticStep(initialInc=1e-03, maxNumInc=10000, minInc=
    1e-5, maxInc=1e-01, name='Step-Pressure', nlgeom=ON, previous='Step-Gravity')

# define FluidCavityProperty
mdb.models['Model-1'].FluidCavityProperty(expansionTable=((1.0, ), ), 
    fluidDensity=1.225e-12, name='IntProp-FluidCavity', useExpansion=True)

mdb.models['Model-1'].FluidCavity(cavityPoint=
    mdb.models['Model-1'].rootAssembly.sets['Set-RP'], cavitySurface=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].surfaces['Surf-Inner Cavity']
    , createStepName='Initial', interactionProperty='IntProp-FluidCavity', 
    name='Int-FluidCavity')

mdb.models['Model-1'].FluidCavityPressureBC(amplitude=UNSET, createStepName=
    'Initial', fluidCavity='Int-FluidCavity', magnitude=0.0, name='BC-FluidCavity')
mdb.models['Model-1'].boundaryConditions['BC-FluidCavity'].setValuesInStep(magnitude=
    P, stepName='Step-Pressure')

# Contact
mdb.models['Model-1'].ContactProperty('IntProp-FaceContact')
mdb.models['Model-1'].interactionProperties['IntProp-FaceContact'].TangentialBehavior(
    formulation=FRICTIONLESS)
mdb.models['Model-1'].rootAssembly.Surface(name='Surf-FaceContact-1', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(((
    -48.428617, 7.96376, 111.666667), ), ((-49.428617, 3.963761, 118.333333), 
    ), ))
mdb.models['Model-1'].rootAssembly.Surface(name='Surf-FaceContact-2', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(((
    -39.428617, 7.96376, 111.666667), ), ((-40.428617, 3.963761, 118.333333), 
    ), ))
mdb.models['Model-1'].rootAssembly.Surface(name='Surf-FaceContact-3', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(((
    -30.428617, 7.96376, 111.666667), ), ((-31.428617, 3.963761, 118.333333), 
    ), ))
mdb.models['Model-1'].rootAssembly.Surface(name='Surf-FaceContact-4', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(((
    -21.428617, 7.96376, 111.666667), ), ((-22.428617, 3.963761, 118.333333), 
    ), ))
mdb.models['Model-1'].rootAssembly.Surface(name='Surf-FaceContact-5', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(((
    -12.428617, 7.96376, 111.666667), ), ((-13.428617, 3.963761, 118.333333), 
    ), ))
mdb.models['Model-1'].rootAssembly.Surface(name='Surf-FaceContact-6', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(((
    -3.428617, 7.96376, 111.666667), ), ((-4.428617, 3.963761, 118.333333), ), 
    ))
mdb.models['Model-1'].rootAssembly.Surface(name='Surf-FaceContact-7', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(((
    5.571383, 7.96376, 111.666667), ), ((4.571383, 3.963761, 118.333333), ), ))
mdb.models['Model-1'].rootAssembly.Surface(name='Surf-FaceContact-8', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(((
    14.571383, 7.96376, 111.666667), ), ((13.571383, 3.963761, 118.333333), ), 
    ))
mdb.models['Model-1'].rootAssembly.Surface(name='Surf-FaceContact-9', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(((
    23.571383, 7.96376, 111.666667), ), ((22.571383, 3.963761, 118.333333), ), 
    ))
mdb.models['Model-1'].rootAssembly.Surface(name='Surf-FaceContact-10', side1Faces=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].faces.findAt(((
    32.571383, 7.96376, 111.666667), ), ((31.571383, 3.963761, 118.333333), ), 
    ))
mdb.models['Model-1'].SelfContactStd(createStepName='Step-Pressure', 
    interactionProperty='IntProp-FaceContact', name='Int-FaceContact-1', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['Surf-FaceContact-1'], thickness=ON)
mdb.models['Model-1'].SelfContactStd(createStepName='Step-Pressure', 
    interactionProperty='IntProp-FaceContact', name='Int-FaceContact-2', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['Surf-FaceContact-2'], thickness=ON)
mdb.models['Model-1'].SelfContactStd(createStepName='Step-Pressure', 
    interactionProperty='IntProp-FaceContact', name='Int-FaceContact-3', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['Surf-FaceContact-3'], thickness=ON)
mdb.models['Model-1'].SelfContactStd(createStepName='Step-Pressure', 
    interactionProperty='IntProp-FaceContact', name='Int-FaceContact-4', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['Surf-FaceContact-4'], thickness=ON)
mdb.models['Model-1'].SelfContactStd(createStepName='Step-Pressure', 
    interactionProperty='IntProp-FaceContact', name='Int-FaceContact-5', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['Surf-FaceContact-5'], thickness=ON)
mdb.models['Model-1'].SelfContactStd(createStepName='Step-Pressure', 
    interactionProperty='IntProp-FaceContact', name='Int-FaceContact-6', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['Surf-FaceContact-6'], thickness=ON)
mdb.models['Model-1'].SelfContactStd(createStepName='Step-Pressure', 
    interactionProperty='IntProp-FaceContact', name='Int-FaceContact-7', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['Surf-FaceContact-7'], thickness=ON)
mdb.models['Model-1'].SelfContactStd(createStepName='Step-Pressure', 
    interactionProperty='IntProp-FaceContact', name='Int-FaceContact-8', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['Surf-FaceContact-8'], thickness=ON)
mdb.models['Model-1'].SelfContactStd(createStepName='Step-Pressure', 
    interactionProperty='IntProp-FaceContact', name='Int-FaceContact-9', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['Surf-FaceContact-9'], thickness=ON)
mdb.models['Model-1'].SelfContactStd(createStepName='Step-Pressure', 
    interactionProperty='IntProp-FaceContact', name='Int-FaceContact-10', surface=
    mdb.models['Model-1'].rootAssembly.surfaces['Surf-FaceContact-10'], thickness=ON)

# mesh
mdb.models['Model-1'].parts['Merged'].setMeshControls(elemShape=TET, regions=
    mdb.models['Model-1'].parts['Merged'].cells, technique=FREE)
mdb.models['Model-1'].parts['Merged'].setElementType(elemTypes=(ElemType(
    elemCode=C3D20R, elemLibrary=STANDARD), ElemType(elemCode=C3D15, 
    elemLibrary=STANDARD), ElemType(elemCode=C3D10, elemLibrary=STANDARD)), 
    regions=(mdb.models['Model-1'].parts['Merged'].cells, ))
mdb.models['Model-1'].parts['Merged'].seedPart(deviationFactor=0.1, 
    minSizeFactor=0.1, size=Meshsize)
mdb.models['Model-1'].parts['Merged'].generateMesh()
mdb.models['Model-1'].parts['Merged'].setElementType(elemTypes=(ElemType(
    elemCode=C3D10H, elemLibrary=STANDARD), ElemType(elemCode=C3D10H, 
    elemLibrary=STANDARD), ElemType(elemCode=C3D10H, elemLibrary=STANDARD)), 
    regions=(mdb.models['Model-1'].parts['Merged'].cells.findAt(((10.73805, 
    -4.369573, 131.0), ), ((49.071383, -6.369572, 120.333333), ), ((32.571383, 
    7.96376, 111.666667), ), ), ))
mdb.models['Model-1'].parts['Merged'].setElementType(elemTypes=(ElemType(
    elemCode=STRI65, elemLibrary=STANDARD), ElemType(elemCode=STRI65, 
    elemLibrary=STANDARD), ElemType(elemCode=STRI65, elemLibrary=STANDARD)), 
    regions=(mdb.models['Model-1'].parts['Merged'].faces.findAt(((10.73805,
    -5.036239, 120.333333),), ), ))

# define history-output:P--V
mdb.models['Model-1'].HistoryOutputRequest(createStepName='Step-Pressure', 
    name='H-Output-2', rebar=EXCLUDE, region=
    mdb.models['Model-1'].rootAssembly.sets['Set-RP'], sectionPoints=DEFAULT, 
    variables=('PCAV', 'CVOL'))
mdb.models['Model-1'].rootAssembly.regenerate()

# define history-output:COORD
mdb.models['Model-1'].rootAssembly.Set(name='Set-Whole body', nodes=
    mdb.models['Model-1'].rootAssembly.instances['Merged-1'].nodes)
mdb.models['Model-1'].HistoryOutputRequest(createStepName='Step-Pressure',
    name='H-Output-3', rebar=EXCLUDE, region=
    mdb.models['Model-1'].rootAssembly.sets['Set-Whole body'], sectionPoints=DEFAULT, 
    variables=('COOR1', 'COOR2', 'COOR3'))
mdb.models['Model-1'].HistoryOutputRequest(createStepName='Step-Pressure', 
    name='H-Output-4', region=MODEL, rebar=EXCLUDE,sectionPoints=DEFAULT,
    variables=('CFNM', ))

# -*- coding: mbcs -*-
from odbAccess import *
import sys
import math
import numpy as np
import os
import csv

stepName = 'Step-Pressure'
jobName = 'Fpn_bending_Gravity' + '_P_' + str(P).replace('.', '')
menuName = 'Fpn_bending_Gravity'
DirName = menuName + '_P_' + str(P)

# define file path
curr_dir = os.getcwd()
training_set_dir = os.path.join(curr_dir, 'FEModelFiles_training_set')
test_set_dir = os.path.join(curr_dir, 'FEModelFiles_test_set')
training_csv_dir = os.path.join(curr_dir, 'FEModelFiles_training_csv')
test_csv_dir = os.path.join(curr_dir, 'FEModelFiles_test_csv')

if P in test_P_values:
    dir_path = os.path.join(test_set_dir, DirName)
    csv_dir = os.path.join(test_csv_dir, DirName)
else:
    dir_path = os.path.join(training_set_dir, DirName)
    csv_dir = os.path.join(training_csv_dir, DirName)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
os.chdir(dir_path)


# save and submit job
mdb.Job(model='Model-1', name=jobName, numCpus=12, numDomains=12,
    userSubroutine='F:/Fast-pn-actuator/Fpn_bending/noBoard/Gravity/uhyper_gent.for')
mdb.saveAs(pathName=jobName)
mdb.jobs[jobName].submit()
mdb.jobs[jobName].waitForCompletion()

odb = openOdb(path=jobName + '.odb')
instanceName = 'MERGED-1'
n_frames = len(odb.steps[stepName].frames)
last_frame = odb.steps[stepName].frames[n_frames-1]


# output PV data
region = 'Node ASSEMBLY.1'
# print(odb.steps[stepName].historyRegions.keys())   You can use this code to check your 'region' name

data = []
volumeValues = np.array(odb.steps[stepName].historyRegions[region].historyOutputs['CVOL'].data)
pressureValues = np.array(odb.steps[stepName].historyRegions[region].historyOutputs['PCAV'].data)
for i in range(len(volumeValues)):
    data.append([odb.steps[stepName].historyRegions[region].historyOutputs['CVOL'].data[i][1],
        odb.steps[stepName].historyRegions[region].historyOutputs['PCAV'].data[i][1]])
data = np.array(data)

current_dir = os.getcwd()
output_file = os.path.join(csv_dir, DirName +'_PV' + '.csv')
np.savetxt(output_file, data, delimiter=',', fmt='%.10e', header='Volume, Pressure', comments='')



# Output contact force data
step = odb.steps[stepName]
for i in range(1, 11):
    try:
        # 1. Build exact region name
        region_name = "NodeSet  Z%06d" % i  #creat NodeSet  Z000001
        region = step.historyRegions[region_name]
        
        # 2. Build exact history key
        hist_key = "CFNM     ASSEMBLY_SURF-FACECONTACT-%d/ASSEMBLY_SURF-FACECONTACT-%d" % (i, i)
        
        # 3. Get data
        cfnm_output = region.historyOutputs[hist_key]
        data = cfnm_output.data
        
        # 4. Save to CSV
        output_filename = os.path.join(csv_dir, "%s_CFNM_%d.csv" % (DirName, i))
        with open(output_filename, 'w') as f:
            f.write("Time,CFNM\n")
            for t, value in data:
                f.write("%.6f,%.6e\n" % (t, value))
        print("Contact face %d CFNM data saved to: %s" % (i, output_filename))
        
    except KeyError as e:
        print("Warning: Missing data for contact pair %d (%s)" % (i, str(e)))



# output node data
def preprocess_element_types(odb):
    node_element_types = {}
    for instance in odb.rootAssembly.instances.values():
        for element in instance.elements:
            el_type = element.type
            for node_label in element.connectivity:
                if node_label not in node_element_types:
                    node_element_types[node_label] = set()
                node_element_types[node_label].add(el_type)
    return node_element_types


def export_node_data(odb, node_element_types):
    last_frame = odb.steps[stepName].frames[-1]
    Disp = last_frame.fieldOutputs['U']
    S = last_frame.fieldOutputs['S']

    output_filename = os.path.join(csv_dir, DirName + '_last_frame_node.csv')
    
    with open(output_filename, 'w') as outfile:
        writer = csv.writer(outfile, lineterminator='\n')
        header = [
            'Node Label', 'X', 'Y', 'Z', 'ElemTypes',
            'U', 'Ux', 'Uy', 'Uz',
            'S11', 'S22', 'S33', 'S12',
            'maxPrincipal', 'midPrincipal', 'minPrincipal', 'mises',
            'StressSource'
        ]
        writer.writerow(header)


        instance = odb.rootAssembly.instances['MERGED-1']
        for node in instance.nodes:
            node_label = node.label
            node_id = node_label

            try:
                region = odb.steps[stepName].historyRegions['Node MERGED-1.%d' % node_label]
                X = region.historyOutputs['COOR1'].data[-1][1]
                Y = region.historyOutputs['COOR2'].data[-1][1]
                Z = region.historyOutputs['COOR3'].data[-1][1]
            except KeyError:
                X, Y, Z = None, None, None

            disp_subset = Disp.getSubset(region=node)
            U = Ux = Uy = Uz = None
            if disp_subset.values:
                disp_values = disp_subset.values[0].data
                Ux, Uy, Uz = disp_values
                U = (Ux**2 + Uy**2 + Uz**2)**0.5

            stress_data = {
                'S11': 0.0, 'S22': 0.0, 'S33': 0.0, 'S12': 0.0,
                'maxPrincipal': 0.0, 'midPrincipal': 0.0, 
                'minPrincipal': 0.0, 'mises': 0.0
            }
            count = 0
            stress_source = 'N/A'

            elem_types = node_element_types.get(node_id, set())

            if 'STRI65' in elem_types:
                try:
                    S_values = S.getSubset(
                        position=INTEGRATION_POINT,
                        region=node
                    ).values
                        
                    for s_val in S_values:
                        sp = s_val.sectionPoint
                        if 'TOP' in sp.description.upper():
                            stress_data['S11'] += s_val.data[0]
                            stress_data['S22'] += s_val.data[1]
                            stress_data['S33'] += s_val.data[2]
                            stress_data['S12'] += s_val.data[3]
                            stress_data['maxPrincipal'] += s_val.maxPrincipal
                            stress_data['midPrincipal'] += s_val.midPrincipal
                            stress_data['minPrincipal'] += s_val.minPrincipal
                            stress_data['mises'] += s_val.mises
                            count += 1
                        elif 'BOTTOM' in sp.description.upper():
                            pass
                        
                    stress_source = 'Shell_TOP'
                        
                except KeyError as e:
                    print('Error processing MERGED-1.%d' % node_label)

            elif 'C3D10H' in elem_types:
                S_values = S.getSubset(
                    position=ELEMENT_NODAL,
                    region=node
                ).values
                    
                for s_val in S_values:
                    stress_data['S11'] += s_val.data[0]
                    stress_data['S22'] += s_val.data[1]
                    stress_data['S33'] += s_val.data[2]
                    stress_data['S12'] += s_val.data[3]
                    stress_data['maxPrincipal'] += s_val.maxPrincipal
                    stress_data['midPrincipal'] += s_val.midPrincipal
                    stress_data['minPrincipal'] += s_val.minPrincipal
                    stress_data['mises'] += s_val.mises
                    count += 1
                stress_source = 'Solid_Avg'

            elif 'F3D3' in elem_types:
                stress_source = 'Surface_NoStress'

            if count > 0:
                for key in stress_data:
                    stress_data[key] /= count


            row = [
                node_label, X, Y, Z, ','.join(elem_types),
                U, Ux, Uy, Uz,
                stress_data['S11'], stress_data['S22'],
                stress_data['S33'], stress_data['S12'],
                stress_data['maxPrincipal'], stress_data['midPrincipal'],
                stress_data['minPrincipal'], stress_data['mises'],
                stress_source
            ]
            writer.writerow(row)


if __name__ == '__main__':
    odb = openOdb(jobName + '.odb')
    node_element_map = preprocess_element_types(odb)
    export_node_data(odb, node_element_map)

# output element data
output_elem_filename = os.path.join(csv_dir, DirName + '_last_frame_elem.csv')
instance = odb.rootAssembly.instances['MERGED-1']

max_nodes = 0
for element in instance.elements:
    if len(element.connectivity) > max_nodes:
        max_nodes = len(element.connectivity)

header = ['Element Label'] + ['Node ' + str(i+1) for i in range(max_nodes)]
with open(output_elem_filename, 'w') as outfile:
    writer = csv.writer(outfile, lineterminator='\n')
    writer.writerow(header)
    for element in instance.elements:
        elem_label = element.label
        connectivity = list(element.connectivity)
        while len(connectivity) < max_nodes:
            connectivity.append(None)
        writer.writerow([elem_label] + connectivity)



import csv
import os

def add_force_column(csv_dir, DirName, contact_map, reaction_nodes):
    node_file = os.path.join(csv_dir, "{}_last_frame_node.csv".format(DirName))
    
    with open(node_file, 'rU') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    
    force_dict = {}
    
    for contact_id in contact_map:
        cfnm_file = os.path.join(csv_dir, "{}_CFNM_{}.csv".format(DirName, contact_id))
        try:
            with open(cfnm_file, 'r') as f:
                last_line = list(csv.reader(f))[-1]
                total_force = float(last_line[1])
                
            nodes = []
            for face_nodes in contact_map[contact_id].values():
                nodes.extend([str(n) for n in face_nodes])
                
            if nodes:
                force_per_node = total_force / len(nodes)
                for node in nodes:
                    force_dict[node] = force_dict.setdefault(node, 0.0) + force_per_node
        except FileNotFoundError:
            print("Warning: lack CF file {cfnm_file}")


    reaction_file = os.path.join(csv_dir, "{}_reaction_forces.csv".format(DirName))
    try:
        with open(reaction_file, 'rU') as f:
            last_line = list(csv.reader(f))[-1]
            total_reaction = float(last_line[1])
            
        if reaction_nodes:
            force_per_node = total_reaction / len(reaction_nodes)
            for node in map(str, reaction_nodes):
                force_dict[node] = force_dict.setdefault(node, 0.0) + force_per_node
    except FileNotFoundError:
        print("Warning: lack RF file {reaction_file}")

    new_header = header + ['F']
    new_rows = []
    for row in rows:
        node_label = row[0]
        force_value = force_dict.get(node_label, 0.0)
        formatted_value = "{:f}".format(force_value)
        new_row = row + [formatted_value]
        new_rows.append(new_row)
    

    backup_file = node_file.replace('.csv', '_backup.csv')
    os.rename(node_file, backup_file)
    
    with open(node_file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_rows)
    print("new file create successfully")

CONTACT_NODE_MAP = {
    1: {
        "face1": [2846, 2847, 2840, 2841],
        "face2": [2827, 2830, 2828, 2831]
    },
    2: {
        "face1": [2798, 2799, 2792, 2793],
        "face2": [2779, 2782, 2780, 2783]
    },
    3: {
        "face1": [2750, 2751, 2744, 2745],
        "face2": [2731, 2734, 2732, 2735]
    },
    4: {
        "face1": [2702, 2703, 2696, 2697],
        "face2": [2683, 2686, 2684, 2687]
    },
    5: {
        "face1": [2654, 2655, 2648, 2649],
        "face2": [2635, 2638, 2636, 2639]
    },
    6: {
        "face1": [2606, 2607, 2600, 2601],
        "face2": [2587, 2590, 2588, 2591]
    },
    7: {
        "face1": [2558, 2559, 2552, 2553],
        "face2": [2539, 2542, 2540, 2543]
    },
    8: {
        "face1": [2510, 2511, 2504, 2505],
        "face2": [2491, 2494, 2492, 2495]
    },
    9: {
        "face1": [2462, 2463, 2456, 2457],
        "face2": [2443, 2446, 2444, 2447]
    },
    10: {
        "face1": [2414, 2415, 2408, 2409],
        "face2": [2395, 2398, 2396, 2399]
    },
}

if __name__ == "__main__":
    add_force_column(
        csv_dir,
        DirName,
        contact_map=CONTACT_NODE_MAP
    )


odb.close()