# pinn-solves-darcy
 The Darcy's equation describes steady - state fluid flow in porous media.
 Combined with the mass conservation equation, it can be expressed as:
 ∇ · (k∇p) = 0   in Ω, where k is the permeability and p is the pressure.
 Assuming the permeability k = 1, the equation simplifies to the Laplacian equation ∇²p = 0   in Ω
 Boundary conditions are set as follows:
 - Left boundary x = 0: p = 1
 - Right boundary x = 1: p = 0
 - Top and bottom boundaries y = 0, y = 1: ∂p/∂n = 0
![image](https://github.com/user-attachments/assets/9838e171-321e-4b95-bd67-3a77dcddbb74)
 - 
![image](https://github.com/user-attachments/assets/d6d2a9de-f306-4beb-94e7-f2dcf376665b)
