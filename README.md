
#  **Accelerated Primal-Dual Methods for Convex-Strongly-Concave Saddle Point Problems** 
Authors: Mohammad Khalafi, Digvijay Boob

This repository presents different versions of our ALPD method on two main problems 
 - $\ell_p$-norm penalty problems with linear constraints: we have the corresponding files for this problem
    - The main file called "main_file_quad" with the following functions
        - ALPD_quad_obj_linear_cons: ALPD method without linearization in dual
        - ALPD_quad_obj_linear_cons_lg: ALPD method with linearization in dual
        - LPD_quad_obj_linear_cons: Standard linearized Primal-Dual (Chambolle-Pock) method with our new step-size policy
- Quadratically constrained quadratic program (QCQP): we have the corresponding files for this problem
   - The main file called "main_file_quad_objective_quad_const" with the following functions
      - ALPD_inexact_quad_obj_quad_cons: Inecaxt ALPD method without linearization in dual
      - ALPD_quad_obj_quad_cons: ALPD method without linearization in dual
      - ALPD_inexact_lg_quad_obj_quad_cons: Inecaxt ALPD method with linearization in dual
      - ALPD_quad_obj_quad_cons_lg: ALPD method with linearization in dual
  ** For for information see [our paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=HA-GlnkAAAAJ&citation_for_view=HA-GlnkAAAAJ:9yKSN-GCB0IC) ** 
   

##  Installation
You need to install CVX. [See here](https://cvxr.com/cvx/)
