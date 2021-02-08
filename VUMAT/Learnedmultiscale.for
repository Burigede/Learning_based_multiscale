      module class_strain
            implicit none 
            save
            integer  :: pipi
            real     :: pipipi
            integer  :: iter_control = 0
            real, dimension (201,100) :: xpca_W1
            real, dimension (201,100) :: xpca_W2
            real, dimension (201,100) :: xpca_W3 
            real, dimension (201,100) :: xpca_W4
            real, dimension (201,100) :: xpca_W5
            real, dimension (201,100) :: xpca_W6 

            real, dimension (201,100) :: ypca_W1 
            real, dimension (201,100) :: ypca_W2 
            real, dimension (201,100) :: ypca_W3 
            real, dimension (201,100) :: ypca_W4 
            real, dimension (201,100) :: ypca_W5 
            real, dimension (201,100) :: ypca_W6

            type strain_path
                  INTEGER :: num_step
                  INTEGER :: nblock
                  real(8), DIMENSION(:,:), allocatable :: e11_path  ! NxM matrix N for each block m for total steps
                  real(8), DIMENSION(:,:), allocatable :: e22_path 
                  real(8), DIMENSION(:,:), allocatable :: e12_path                      
                  real(8), DIMENSION(:,:), allocatable :: e13_path                                     
                  real(8), DIMENSION(:,:), allocatable :: e23_path                                     
                  real(8), DIMENSION(:,:), allocatable :: e33_path  
                                                     
               
                  end type strain_path

            
            type stress_path
                  INTEGER :: num_step
                  INTEGER :: nblock
                  real(8), DIMENSION(:,:), allocatable :: sig11_path  ! NxM matrix N for each block m for total steps
                  real(8), DIMENSION(:,:), allocatable :: sig22_path 
                  real(8), DIMENSION(:,:), allocatable :: sig12_path  
                  real(8), DIMENSION(:,:), allocatable :: sig33_path  
                  real(8), DIMENSION(:,:), allocatable :: sig13_path  
                  real(8), DIMENSION(:,:), allocatable :: sig23_path  

                  end type stress_path

            type PCA_type
                  integer :: yin_width
                  integer :: xout_width
                  real(8), DIMENSION(:,:), allocatable :: xPCA_W1
                  real(8), DIMENSION(:,:), allocatable :: xPCA_W2
                  real(8), DIMENSION(:,:), allocatable :: xPCA_W3

                  real(8), DIMENSION(:,:), allocatable :: yPCA_W1
                  real(8), DIMENSION(:,:), allocatable :: yPCA_W2
                  real(8), DIMENSION(:,:), allocatable :: yPCA_W3
                  real(8), DIMENSION(:,:), allocatable :: yPCA_W4
                  end type PCA_type        
            contains

      end module class_strain

      module class_layer 
            implicit none 
            save
            real, dimension (1000,600) :: weight1
            real, dimension (1000) :: bias1
            
            real, dimension (1000,1000) :: weight2
            real, dimension (1000) :: bias2

            real, dimension (1000,1000) :: weight3
            real, dimension (1000) :: bias3

            real, dimension (1000,1000) :: weight4
            real, dimension (1000) :: bias4

            real, dimension (600,1000) :: weight5
            real, dimension (600) :: bias5

            real(8) :: scale = 1.0507009873554804934193349852946  ! https://pytorch.org/docs/stable/nn.html selu! 
            real(8) :: alpha = 1.6732632423543772848170429916717 

      end module class_layer 


      module class_net
            use class_strain
            use class_layer
            implicit none 
            save
            contains 

            SUBROUTINE forward(strain_,stress_)
                  type(strain_path), intent(in) :: strain_ 
                  type(stress_path), intent(out) :: stress_

                  real(8) :: net_input_1(strain_%nblock,100)
                  real(8) :: net_input_2(strain_%nblock,100)
                  real(8) :: net_input_3(strain_%nblock,100)
                  real(8) :: net_input_4(strain_%nblock,100)
                  real(8) :: net_input_5(strain_%nblock,100)
                  real(8) :: net_input_6(strain_%nblock,100)


                  real(8) :: net_input(600)

                  real(8) :: net_output_1(strain_%nblock,100)
                  real(8) :: net_output_2(strain_%nblock,100)
                  real(8) :: net_output_3(strain_%nblock,100)
                  real(8) :: net_output_4(strain_%nblock,100)
                  real(8) :: net_output_5(strain_%nblock,100)
                  real(8) :: net_output_6(strain_%nblock,100)


                  real(8) :: layer_1_out(1000)
                  real(8) :: layer_2_out(1000)
                  real(8) :: layer_3_out(1000)
                  real(8) :: layer_4_out(1000)
                  real(8) :: layer_5_out(600)

                  real(8) :: net_output(600)
                  integer :: d1,d2, i,j

                  d1 = 600
                  d2 = 600

                  allocate(stress_%sig11_path(strain_%nblock,strain_%num_step))
                  allocate(stress_%sig22_path(strain_%nblock,strain_%num_step))
                  allocate(stress_%sig33_path(strain_%nblock,strain_%num_step))
                  allocate(stress_%sig12_path(strain_%nblock,strain_%num_step))
                  allocate(stress_%sig13_path(strain_%nblock,strain_%num_step))
                  allocate(stress_%sig23_path(strain_%nblock,strain_%num_step))

         
                  net_input_1 =  MATMUL(strain_%e11_path,xPCA_W1)
                  net_input_2 =  MATMUL(strain_%e22_path,xPCA_W2)
                  net_input_3 =  MATMUL(strain_%e33_path,xPCA_W3) 
                  net_input_4 =  MATMUL(strain_%e12_path,xPCA_W4)
                  net_input_5 =  MATMUL(strain_%e23_path,xPCA_W5)
                  net_input_6 =  MATMUL(strain_%e13_path,xPCA_W6)

                  do i = 1,strain_%nblock
                        !print *, "i", i
                        net_input(1:d1:6)=reshape(net_input_1(i,:),(/100/))
                        net_input(2:d1:6)=reshape(net_input_2(i,:),(/100/))
                        net_input(3:d1:6)=reshape(net_input_3(i,:),(/100/)) 
                        net_input(4:d1:6)=reshape(net_input_4(i,:),(/100/)) 
                        net_input(5:d1:6)=reshape(net_input_5(i,:),(/100/)) 
                        net_input(6:d1:6)=reshape(net_input_6(i,:),(/100/)) 

                        layer_1_out    = MATMUL(weight1, net_input) + bias1  

                        do j = 1, 1000 
                            layer_1_out(j) = scale * (max(0d0, layer_1_out(j))+min(0d0,alpha*(exp(layer_1_out(j))-1d0)))
                        enddo  
                        
                        layer_2_out    = MATMUL(weight2, layer_1_out) + bias2  

                        do j = 1, 1000 
                            layer_2_out(j) = scale * (max(0d0, layer_2_out(j))+min(0d0,alpha*(exp(layer_2_out(j))-1d0)))
                        enddo  

                        layer_3_out    = MATMUL(weight3, layer_2_out) + bias3  

                        do j = 1, 1000 
                            layer_3_out(j) = scale * (max(0d0, layer_3_out(j))+min(0d0,alpha*(exp(layer_3_out(j))-1d0)))
                        enddo  

                        layer_4_out    = MATMUL(weight4, layer_3_out) + bias4  

                        do j = 1, 1000 
                            layer_4_out(j) = scale * (max(0d0, layer_4_out(j))+min(0d0,alpha*(exp(layer_4_out(j))-1d0)))
                        enddo  

                        layer_5_out    = MATMUL(weight5, layer_4_out) + bias5  

                        net_output  = layer_5_out 

                        net_output_1(i,:) = net_output(1:d2:6)
                        net_output_2(i,:) = net_output(2:d2:6)
                        net_output_3(i,:) = net_output(3:d2:6)
                        net_output_4(i,:) = net_output(4:d2:6)
                        net_output_5(i,:) = net_output(5:d2:6)
                        net_output_6(i,:) = net_output(6:d2:6)
 
                  enddo
                  stress_%sig11_path = MATMUL(net_output_1,transpose(yPCA_W1))
                  stress_%sig22_path = MATMUL(net_output_2,transpose(yPCA_W2))
                  stress_%sig33_path = MATMUL(net_output_3,transpose(yPCA_W3))
                  stress_%sig12_path = MATMUL(net_output_4,transpose(yPCA_W4))
                  stress_%sig23_path = MATMUL(net_output_5,transpose(yPCA_W5))
                  stress_%sig13_path = MATMUL(net_output_6,transpose(yPCA_W6))
            
            end SUBROUTINE forward
            
      end module class_net

      subroutine vumat(
C Read only (unmodifiable)variables -
     1  nblock, ndir, nshr, nstatev, nfieldv, nprops, lanneal,
     2  stepTime, totalTime, dt, cmname, coordMp, charLength,
     3  props, density, strainInc, relSpinInc,
     4  tempOld, stretchOld, defgradOld, fieldOld,
     5  stressOld, stateOld, enerInternOld, enerInelasOld,
     6  tempNew, stretchNew, defgradNew, fieldNew,
C Write only (modifiable) variables -
     7  stressNew, stateNew, enerInternNew, enerInelasNew )
C
      use class_strain
      use class_layer  
      use class_net
      include 'vaba_param.inc'
C
      dimension props(nprops), density(nblock), coordMp(nblock,*),
     1  charLength(nblock), strainInc(nblock,ndir+nshr),
     2  relSpinInc(nblock,nshr), tempOld(nblock),
     3  stretchOld(nblock,ndir+nshr),
     4  defgradOld(nblock,ndir+nshr+nshr),
     5  fieldOld(nblock,nfieldv), stressOld(nblock,ndir+nshr),
     6  stateOld(nblock,nstatev), enerInternOld(nblock),
     7  enerInelasOld(nblock), tempNew(nblock),
     8  stretchNew(nblock,ndir+nshr),
     8  defgradNew(nblock,ndir+nshr+nshr),
     9  fieldNew(nblock,nfieldv),
     1  stressNew(nblock,ndir+nshr), stateNew(nblock,nstatev),
     2  enerInternNew(nblock), enerInelasNew(nblock)
C
      character*80 cmname
C

      dimension ep3(3,3),dep3(3,3),et3(3,3),det3(3,3),xiden(3,3),
     *  sig3(3,3), f(3,3), sigma_dev(3,3), e3(3,3), strain_elas(3,3)

      parameter(zero=0d0,one=1d0,half=0.5d0,two=2d0,three=3d0)  
      INTEGER :: in_width, out_width, n_layers
      real(8), DIMENSION(2)   :: input
      real(8), DIMENSION(2)   :: output  
      real(8), DIMENSION(2)   :: bias
      real(8), DIMENSION(2,2) :: weights 
      real(8), DIMENSION(nblock,6) :: F_new
      real(8), DIMENSION(nblock,6) :: F_old


C==============================================================================C
CCC==================== Initialize strain path ================================C
      INTEGER :: total_steps
      type(strain_path) :: strain_all
      type(stress_path) :: stress_all
CCC==================== Initialize PCA ======================================CCC
      !type(PCA_type) :: PCA_all  

      !type(layer) :: xlayer_1, xlayer_2, xlayer_3, xlayer_4, xlayer_5
      !type(net)   :: test_net
      integer     :: iter
      integer     :: n_element=225,max_block=136
      integer     :: block_k=1
      character(len=5) :: charI
      !save strain_all 
      !save iter

      xiden(1,1)=one
      xiden(2,1)=zero
      xiden(3,1)=zero 
      xiden(1,2)=zero
      xiden(2,2)=one
      xiden(3,2)=zero
      xiden(1,3)=zero
      xiden(2,3)=zero
      xiden(3,3)=one   

      !print *, "iter", iter  

      total_steps = 201
      
      
        strain_all%num_step = total_steps
        strain_all%nblock = nblock
        allocate(strain_all%e11_path(nblock,total_steps))
        allocate(strain_all%e22_path(nblock,total_steps))
        allocate(strain_all%e33_path(nblock,total_steps))
        allocate(strain_all%e12_path(nblock,total_steps))
        allocate(strain_all%e23_path(nblock,total_steps))
        allocate(strain_all%e13_path(nblock,total_steps))
   

        if (iter_control.eq.zero) then
            open (1, file = 'xpca_Taylor_W1.txt', status = 'old')
            read(1,*) xpca_W1
            close(1) 

            open (2, file = 'xpca_Taylor_W2.txt', status = 'old')
            read(2,*) xPCA_W2
            close(2) 

            open (3, file = 'xpca_Taylor_W3.txt', status = 'old')
            read(3,*) xPCA_W3
            close(3) 

            open (4, file = 'xpca_Taylor_W4.txt', status = 'old')
            read(4,*) xPCA_W4
            close(4) 

            open (5, file = 'xpca_Taylor_W5.txt', status = 'old')
            read(5,*) xPCA_W5
            close(5) 

            open (6, file = 'xpca_Taylor_W6.txt', status = 'old')
            read(6,*) xPCA_W6
            close(6) 

            open (7, file = 'ypca_Taylor_W1.txt', status = 'old')
            read(7,*) yPCA_W1
            close(7) 

            open (8, file = 'ypca_Taylor_W2.txt', status = 'old')
            read(8,*) yPCA_W2
            close(8) 

            open (9, file = 'ypca_Taylor_W3.txt', status = 'old')
            read(9,*) yPCA_W3
            close(9)

            open (10, file = 'ypca_Taylor_W4.txt', status = 'old')
            read(10,*) yPCA_W4
            close(10) 

            open (10, file = 'ypca_Taylor_W5.txt', status = 'old')
            read(10,*) yPCA_W5
            close(10) 

            open (10, file = 'ypca_Taylor_W6.txt', status = 'old')
            read(10,*) yPCA_W6
            close(10) 

            open (1, file = 'layer_1_Taylor_weight.txt', status = 'old')
            read(1,*) weight1
            close(1) 

            open (2, file = 'layer_1_Taylor_bias.txt', status = 'old')
            read(2,*) bias1
            close(2)

            open (1, file = 'layer_2_Taylor_weight.txt', status = 'old')
            read(1,*) weight2
            close(1) 

            open (2, file = 'layer_2_Taylor_bias.txt', status = 'old')
            read(2,*) bias2
            close(2)  
            
            open (1, file = 'layer_3_Taylor_weight.txt', status = 'old')
            read(1,*) weight3
            close(1) 

            open (2, file = 'layer_3_Taylor_bias.txt', status = 'old')
            read(2,*) bias3
            close(2)  


            open (1, file = 'layer_4_Taylor_weight.txt', status = 'old')
            read(1,*) weight4
            close(1) 

            open (2, file = 'layer_4_Taylor_bias.txt', status = 'old')
            read(2,*) bias4
            close(2)  

            open (1, file = 'layer_5_Taylor_weight.txt', status = 'old')
            read(1,*) weight5
            close(1) 

            open (2, file = 'layer_5_Taylor_bias.txt', status = 'old')
            read(2,*) bias5
            close(2)  

            !print *,"init here"
            !print *, iter 
            iter_control = iter_control + 1 
        endif

        !print *, "iter_control", iter_control
        do km = 1, nblock 
            iter = stateOld(km,7)

            !print *, 'iter_start' , iter
           

            F_new(km,1) = stretchNew(km,1)  
            F_new(km,2) = stretchNew(km,2)
            F_new(km,3) = stretchNew(km,3)
            F_new(km,4) = stretchNew(km,4) 
            F_new(km,5) = stretchNew(km,5) 
            F_new(km,6) = stretchNew(km,6) 

            !print *, stretchNew(km,1)

            if (iter.eq.zero) then
                  F_old(km,1) = 1.0
                  F_old(km,2) = 1.0
                  F_old(km,3) = 1.0
                  F_old(km,4) = 0.0
                  F_old(km,5) = 0.0
                  F_old(km,6) = 0.0
            else 
                  F_old(km,1) = stateOld(km,1)
                  F_old(km,2) = stateOld(km,2)
                  F_old(km,3) = stateOld(km,3)
                  F_old(km,4) = stateOld(km,4)
                  F_old(km,5) = stateOld(km,5)
                  F_old(km,6) = stateOld(km,6)
            endif 

            do j = 1,total_steps
                  if (iter.eq.zero) then
                        strain_all%e11_path(km,j) = 1.0
                        strain_all%e22_path(km,j) = 1.0
                        strain_all%e33_path(km,j) = 1.0
                        strain_all%e12_path(km,j) = 0.0
                        strain_all%e23_path(km,j) = 0.0
                        strain_all%e13_path(km,j) = 0.0
                  else
                        strain_all%e11_path(km,j) = stateOld(km,j+7)
                        strain_all%e22_path(km,j) = stateOld(km,j+7+total_steps)
                        strain_all%e33_path(km,j) = stateOld(km,j+7+2*total_steps)
                        strain_all%e12_path(km,j) = stateOld(km,j+7+3*total_steps)
                        strain_all%e23_path(km,j) = stateOld(km,j+7+4*total_steps)
                        strain_all%e13_path(km,j) = stateOld(km,j+7+5*total_steps) 
                  endif
            enddo

            do j = iter+1, total_steps
                  strain_all%e11_path(km,j) = F_old(km,1)+(F_new(km,1)-F_old(km,1))*(j-(iter+1))
                  strain_all%e22_path(km,j) = F_old(km,2)+(F_new(km,2)-F_old(km,2))*(j-(iter+1))
                  strain_all%e33_path(km,j) = F_old(km,3)+(F_new(km,3)-F_old(km,3))*(j-(iter+1))
                  strain_all%e12_path(km,j) = F_old(km,4)+(F_new(km,4)-F_old(km,4))*(j-(iter+1))
                  strain_all%e23_path(km,j) = F_old(km,5)+(F_new(km,5)-F_old(km,5))*(j-(iter+1))
                  strain_all%e13_path(km,j) = F_old(km,6)+(F_new(km,6)-F_old(km,6))*(j-(iter+1)) 
            enddo 
            
            do j = 1, total_steps
                  stateNew(km,j+7)                 = strain_all%e11_path(km,j)
                  stateNew(km,j+7+total_steps)     = strain_all%e22_path(km,j)
                  stateNew(km,j+7+2*total_steps)   = strain_all%e33_path(km,j)
                  stateNew(km,j+7+3*total_steps)   = strain_all%e12_path(km,j)
                  stateNew(km,j+7+4*total_steps)   = strain_all%e23_path(km,j)
                  stateNew(km,j+7+5*total_steps)   = strain_all%e13_path(km,j)
            enddo 

            iter = iter + 1 
            !print *, "total_steps"
            !!print *, total_steps
            !print *, 'iter_end' , iter
            !print *, "e11"
            !print *, strain_all%e11_path
            stateNew(km,7)  = iter  

            f(1,1) = F_new(km,1) 
            f(2,2) = F_new(km,2) 
            f(3,3) = F_new(km,3) 
            f(1,2) = F_new(km,4) 
            f(2,3) = F_new(km,5) 
            f(1,3) = F_new(km,6) 
            f(2,1) = f(1,2) 
            f(3,2) = f(2,3) 
            f(3,1) = f(1,3) 
            
            detF = f(1,1)*(f(2,2)*f(3,3)-f(2,3)*f(3,2))
            detF = detF - f(1,2)*(f(2,1)*f(3,3)-f(2,3)*f(3,1))
            detF = detF + f(1,3)*(f(2,1)*f(3,2)-f(2,2)*f(3,1))
            
            do mm = 1,3 
              do nn = 1,3 
                  f(mm,nn) = f(mm,nn)/detF
              enddo 
            enddo

            strain_elas = MATMUL(transpose(f), f)  
            strain_elas = half*(strain_elas - xiden) 

            stateNew(km, 1214) = zero
            do m = 1,2
               do n = 1,2
                  stateNew(km, 1214) = stateNew(km, 1214) + strain_elas(m,n)*strain_elas(m,n) 
               enddo 
            enddo
            stateNew(km,1214) = sqrt(stateNew(km,1214))  
            stateNew(km,1215) = detF - one
        enddo

      call  forward(strain_all,stress_all) 

      !open(2, file = 'C:\ABAQUS_WORK\stress11.txt')  
      !do i=1,201 
      !write(2,*)  stress_all%sig11_path(1,i), stress_all%sig22_path(1,i), stress_all%sig12_path(1,i)
      !end do   
       
       !print *, "Fnew", F_new(8,2);
       !print *, "Fold", F_old(8,2);
       !print *, "strain 11"
       !print *, strain_all%e11_path(8,:)
       !print *, "strain 22"
       !print *, strain_all%e22_path(8,:)
       !print *, "strain 12"
       !print *, strain_all%e12_path(8,:)

c        print *, "here" 
        !print *, "e11path" 
        !print *, strain_all%e11_path

        !print *, "e22path" 
        !print *, strain_all%e22_path
     
        
        !print *, "e33path" 
        !print *, strain_all%e33_path

        !print *, "e12path" 
        !print *, strain_all%e12_path
        
        !print *, "e23path" 
        !print *, strain_all%e23_path

        !print *, "e13path" 
        !print *, strain_all%e12_path

        stressNew(:,1)  = 1000.0*stress_all%sig11_path(:,iter+1)
        stressNew(:,2)  = 1000.0*stress_all%sig22_path(:,iter+1)
        stressNew(:,3)  = 1000.0*stress_all%sig33_path(:,iter+1)
        stressNew(:,4)  = 1000.0*stress_all%sig12_path(:,iter+1)
        stressNew(:,5)  = 1000.0*stress_all%sig23_path(:,iter+1)
        stressNew(:,6)  = 1000.0*stress_all%sig13_path(:,iter+1)
       
        !print *, "stressNew" 
        !print *, stressNew

c        print *, "stress 11"
c        print *, stressNew(:,1)

c        print *, "stress 22"
c        print *, stressNew(:,2)
        
c        print *, "stress 12"
c        print *, stressNew(:,4)
        

c       print *, "stress 33"
c        print *, stressNew(:,3)
        


        stateNew(:,1)  = F_new(:,1)
        stateNew(:,2)  = F_new(:,2)
        stateNew(:,3)  = F_new(:,3)
        stateNew(:,4)  = F_new(:,4)
        stateNew(:,5)  = F_new(:,5)
        stateNew(:,6)  = F_new(:,6)
c        print *, "========="
c        print *, "iteration" 
c        print *, iter

      return
      end


