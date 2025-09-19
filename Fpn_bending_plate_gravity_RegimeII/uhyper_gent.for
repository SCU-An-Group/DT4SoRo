      SUBROUTINE UHYPER(BI1,BI2,AJ,U,UI1,UI2,UI3,TEMP,NOEL,
     +  CMNAME,INCMPFLAG,NUMSTATEV,STATEV,NUMFIELDV,FIELDV,
     +  FIELDVINC,NUMPROPS,PROPS)
C
        INCLUDE 'ABA_PARAM.INC'
C
        CHARACTER*80 CMNAME

        DIMENSION U(2),UI1(3),UI2(6),UI3(6),PROPS(NUMPROPS)


C


C       Nick Vasios
C       Harvard J.A.Paulson SEAS
C       Bertoldi Group
C       May 2017

C       Hyperelastic behavior of a material with a GENT strain energy density

C       BI1 is I1 bar
C       BI2 is I2 bar
C       AJ is det(F)

        IWR = 0
        IOUT = 7

        IF (NUMPROPS.LT.2) THEN
          WRITE(IOUT,*) 'LESS THAN TWO MATERIAL PROPERTIES FOUND'
          WRITE(IOUT,*) 'UHYPER ABORTS'
          CALL XIT
        END IF

        AMU = PROPS(1)
        AJM = PROPS(2)

        IF (IWR.EQ.1) THEN
          WRITE(IOUT,*) 'NOW PROCESSING ELEMENT'
          WRITE(IOUT,1002) NOEL
          WRITE(IOUT,*)
          WRITE(IOUT,*) 'MATERIAL PROPERTIES: MU AND JM'
          WRITE(IOUT,1001) AMU,AJM
        END IF

C       THE STRAIN ENERGY DENSITY
        U(1) = -(AMU*AJM/2.D0)*DLOG(1.D0-((AJ**(-2.D0/3.D0))*BI1-3.D0)/AJM)

C       USEFUL PARAMETER
        AUX = BI1 - (AJ**(2.D0/3.D0))*(3.D0 + AJM)

C       FIRST DERIVATIVES OF THE STRAIN ENERGY DENSITY
        AUX1 = AMU*AJM/AUX

        UI1(1) = AUX1*(-0.5D0)
        UI1(2) = 0.D0
        UI1(3) = AUX1*(BI1 / (3.D0 * AJ))

C       SECOND DERIVATIVES
        AUX2 = AMU*AJM/ AUX**2.D0
        AUX3 = (AJ**(2.D0/3.D0))*(3.D0 + AJM)

        UI2(1) = AUX2*0.5D0
        UI2(2) = 0.D0
        UI2(3) = AUX2*(BI1*(-3.D0*BI1 + 5.D0*AUX3))/(9.D0 * AJ**2.D0)
        UI2(4) = 0.D0
        UI2(5) = -AUX2*(3.D0 + AJM)/(3.D0*AJ**(1.D0/3.D0))
        UI2(6) = 0.D0

C       THIRD DERIVATIVES
        AUX4 = AMU*AJM/ AUX**3.D0
        UI3(1) = AUX4*(2.D0*(3.D0 + AJM)/(3.D0 * AJ**(1.D0/3.D0)))
        UI3(2) = 0.D0
        UI3(3) = 0.D0
        UI3(4) = -AUX4*(3.D0+AJM)*(-BI1+5.D0*AUX3)/(9.D0*AJ**(4.D0/3.D0))
        UI3(5) = 0.D0
        UI3(6) = AUX4*2.D0*BI1*(9.D0* BI1**2.D0  -25.D0*AUX3 +
     +  20.D0*AUX3**2.D0)/(27.D0 * AJ**3.D0)


      RETURN

 1001 FORMAT(1P8E13.5)
 1002 FORMAT(10I5)
      END
