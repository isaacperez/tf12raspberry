??
?$?$
:
Add
x"T
y"T
z"T"
Ttype:
2	
?
	ApplyAdam
var"T?	
m"T?	
v"T?
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
?
AvgPoolGrad
orig_input_shape	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
Equal
x"T
y"T
z
"
Ttype:
2	
?
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.11.02v1.11.0-rc2-4-gc19e29306c??
f
XPlaceholder*
dtype0*(
_output_shapes
:??????????*
shape:??????????
f
Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"????         
l
ReshapeReshapeXReshape/shape*
T0*
Tshape0*/
_output_shapes
:?????????
o
truncated_normal/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
dtype0*
seed2 *&
_output_shapes
:*

seed 
?
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*&
_output_shapes
:
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:
?
W_c1
VariableV2*
dtype0*
	container *&
_output_shapes
:*
shape:*
shared_name 
?
W_c1/AssignAssignW_c1truncated_normal*
use_locking(*
T0*
_class
	loc:@W_c1*
validate_shape(*&
_output_shapes
:
e
	W_c1/readIdentityW_c1*
T0*
_class
	loc:@W_c1*&
_output_shapes
:
b
truncated_normal_1/shapeConst*
valueB:*
dtype0*
_output_shapes
:
\
truncated_normal_1/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_1/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*
seed2 *
_output_shapes
:*

seed *
T0
?
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes
:
o
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0*
_output_shapes
:
p
b_c1
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
?
b_c1/AssignAssignb_c1truncated_normal_1*
use_locking(*
T0*
_class
	loc:@b_c1*
validate_shape(*
_output_shapes
:
Y
	b_c1/readIdentityb_c1*
_output_shapes
:*
T0*
_class
	loc:@b_c1
i
truncated_normal_2/shapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *???=
?
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
T0*
dtype0*
seed2 *
_output_shapes

:

?
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
_output_shapes

:
*
T0
s
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*
_output_shapes

:

x
W_n1
VariableV2*
shape
:
*
shared_name *
dtype0*
	container *
_output_shapes

:

?
W_n1/AssignAssignW_n1truncated_normal_2*
use_locking(*
T0*
_class
	loc:@W_n1*
validate_shape(*
_output_shapes

:

]
	W_n1/readIdentityW_n1*
_output_shapes

:
*
T0*
_class
	loc:@W_n1
b
truncated_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB:

\
truncated_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_3/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*
seed2 *
_output_shapes
:
*

seed *
T0
?
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:

o
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:

p
b_n1
VariableV2*
dtype0*
	container *
_output_shapes
:
*
shape:
*
shared_name 
?
b_n1/AssignAssignb_n1truncated_normal_3*
T0*
_class
	loc:@b_n1*
validate_shape(*
_output_shapes
:
*
use_locking(
Y
	b_n1/readIdentityb_n1*
T0*
_class
	loc:@b_n1*
_output_shapes
:

?
Conv2DConv2DReshape	W_c1/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:?????????*
	dilations

S
conv1IdentityConv2D*
T0*/
_output_shapes
:?????????
u
BiasAddBiasAddconv1	b_c1/read*
data_formatNHWC*/
_output_shapes
:?????????*
T0
]
conv1_con_biasIdentityBiasAdd*/
_output_shapes
:?????????*
T0
V
ReluReluconv1_con_bias*
T0*/
_output_shapes
:?????????
`
conv1_con_activacionIdentityRelu*
T0*/
_output_shapes
:?????????
?
AvgPoolAvgPoolconv1_con_activacion*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*/
_output_shapes
:?????????*
T0
S
poolIdentityAvgPool*/
_output_shapes
:?????????*
T0
`
Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
k
	Reshape_1ReshapepoolReshape_1/shape*
T0*
Tshape0*'
_output_shapes
:?????????
O
pool1DIdentity	Reshape_1*'
_output_shapes
:?????????*
T0
{
MatMulMatMulpool1D	W_n1/read*
transpose_a( *'
_output_shapes
:?????????
*
transpose_b( *
T0
I
nn1IdentityMatMul*
T0*'
_output_shapes
:?????????

L
addAddnn1	b_n1/read*
T0*'
_output_shapes
:?????????

I
logitsIdentityadd*
T0*'
_output_shapes
:?????????

L
SoftmaxSoftmaxlogits*
T0*'
_output_shapes
:?????????

n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:?????????
*
shape:?????????

E
LogLogSoftmax*
T0*'
_output_shapes
:?????????

N
mulMulPlaceholderLog*
T0*'
_output_shapes
:?????????

V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
T
SumSummulConst*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
0
NegNegSum*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ??*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
N
gradients/Neg_grad/NegNeggradients/Fill*
_output_shapes
: *
T0
q
 gradients/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
?
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
[
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:
?
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*
T0*'
_output_shapes
:?????????
*

Tmultiples0
c
gradients/mul_grad/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:
]
gradients/mul_grad/Shape_1ShapeLog*
T0*
out_type0*
_output_shapes
:
?
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
m
gradients/mul_grad/MulMulgradients/Sum_grad/TileLog*
T0*'
_output_shapes
:?????????

?
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

w
gradients/mul_grad/Mul_1MulPlaceholdergradients/Sum_grad/Tile*
T0*'
_output_shapes
:?????????

?
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????

g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
?
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*'
_output_shapes
:?????????
*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape
?
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*'
_output_shapes
:?????????

?
gradients/Log_grad/Reciprocal
ReciprocalSoftmax.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:?????????

?
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:?????????

t
gradients/Softmax_grad/mulMulgradients/Log_grad/mulSoftmax*
T0*'
_output_shapes
:?????????

w
,gradients/Softmax_grad/Sum/reduction_indicesConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0*'
_output_shapes
:?????????
?
gradients/Softmax_grad/subSubgradients/Log_grad/mulgradients/Softmax_grad/Sum*
T0*'
_output_shapes
:?????????

z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*'
_output_shapes
:?????????
*
T0
[
gradients/add_grad/ShapeShapenn1*
T0*
out_type0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
?
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/add_grad/SumSumgradients/Softmax_grad/mul_1(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
?
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:?????????
*
T0*
Tshape0
?
gradients/add_grad/Sum_1Sumgradients/Softmax_grad/mul_1*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
?
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
?
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:?????????

?
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:

?
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependency	W_n1/read*
T0*
transpose_a( *'
_output_shapes
:?????????*
transpose_b(
?
gradients/MatMul_grad/MatMul_1MatMulpool1D+gradients/add_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:
*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
?
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:?????????
?
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:

b
gradients/Reshape_1_grad/ShapeShapepool*
_output_shapes
:*
T0*
out_type0
?
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*/
_output_shapes
:?????????*
T0*
Tshape0
p
gradients/AvgPool_grad/ShapeShapeconv1_con_activacion*
T0*
out_type0*
_output_shapes
:
?
"gradients/AvgPool_grad/AvgPoolGradAvgPoolGradgradients/AvgPool_grad/Shape gradients/Reshape_1_grad/Reshape*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:?????????*
T0
?
gradients/Relu_grad/ReluGradReluGrad"gradients/AvgPool_grad/AvgPoolGradRelu*/
_output_shapes
:?????????*
T0
?
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp#^gradients/BiasAdd_grad/BiasAddGrad^gradients/Relu_grad/ReluGrad
?
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:?????????
?
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad
~
gradients/Conv2D_grad/ShapeNShapeNReshape	W_c1/read*
T0*
out_type0*
N* 
_output_shapes
::
?
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeN	W_c1/read/gradients/BiasAdd_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:?????????*
	dilations
*
T0
?
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:*
	dilations

?
&gradients/Conv2D_grad/tuple/group_depsNoOp+^gradients/Conv2D_grad/Conv2DBackpropFilter*^gradients/Conv2D_grad/Conv2DBackpropInput
?
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
w
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
	loc:@W_c1*
dtype0*
_output_shapes
: 
?
beta1_power
VariableV2*
shared_name *
_class
	loc:@W_c1*
	container *
shape: *
dtype0*
_output_shapes
: 
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
	loc:@W_c1*
validate_shape(*
_output_shapes
: 
c
beta1_power/readIdentitybeta1_power*
T0*
_class
	loc:@W_c1*
_output_shapes
: 
w
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w??*
_class
	loc:@W_c1
?
beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
	loc:@W_c1*
	container 
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
	loc:@W_c1*
validate_shape(*
_output_shapes
: 
c
beta2_power/readIdentitybeta2_power*
T0*
_class
	loc:@W_c1*
_output_shapes
: 
?
W_c1/Adam/Initializer/zerosConst*%
valueB*    *
_class
	loc:@W_c1*
dtype0*&
_output_shapes
:
?
	W_c1/Adam
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *
_class
	loc:@W_c1*
	container *
shape:
?
W_c1/Adam/AssignAssign	W_c1/AdamW_c1/Adam/Initializer/zeros*
T0*
_class
	loc:@W_c1*
validate_shape(*&
_output_shapes
:*
use_locking(
o
W_c1/Adam/readIdentity	W_c1/Adam*&
_output_shapes
:*
T0*
_class
	loc:@W_c1
?
W_c1/Adam_1/Initializer/zerosConst*%
valueB*    *
_class
	loc:@W_c1*
dtype0*&
_output_shapes
:
?
W_c1/Adam_1
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *
_class
	loc:@W_c1*
	container *
shape:
?
W_c1/Adam_1/AssignAssignW_c1/Adam_1W_c1/Adam_1/Initializer/zeros*
T0*
_class
	loc:@W_c1*
validate_shape(*&
_output_shapes
:*
use_locking(
s
W_c1/Adam_1/readIdentityW_c1/Adam_1*
T0*
_class
	loc:@W_c1*&
_output_shapes
:
?
b_c1/Adam/Initializer/zerosConst*
valueB*    *
_class
	loc:@b_c1*
dtype0*
_output_shapes
:
?
	b_c1/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
	loc:@b_c1
?
b_c1/Adam/AssignAssign	b_c1/Adamb_c1/Adam/Initializer/zeros*
T0*
_class
	loc:@b_c1*
validate_shape(*
_output_shapes
:*
use_locking(
c
b_c1/Adam/readIdentity	b_c1/Adam*
_output_shapes
:*
T0*
_class
	loc:@b_c1
?
b_c1/Adam_1/Initializer/zerosConst*
valueB*    *
_class
	loc:@b_c1*
dtype0*
_output_shapes
:
?
b_c1/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
	loc:@b_c1*
	container 
?
b_c1/Adam_1/AssignAssignb_c1/Adam_1b_c1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_c1*
validate_shape(*
_output_shapes
:
g
b_c1/Adam_1/readIdentityb_c1/Adam_1*
T0*
_class
	loc:@b_c1*
_output_shapes
:
?
W_n1/Adam/Initializer/zerosConst*
valueB
*    *
_class
	loc:@W_n1*
dtype0*
_output_shapes

:

?
	W_n1/Adam
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *
_class
	loc:@W_n1*
	container *
shape
:

?
W_n1/Adam/AssignAssign	W_n1/AdamW_n1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@W_n1*
validate_shape(*
_output_shapes

:

g
W_n1/Adam/readIdentity	W_n1/Adam*
T0*
_class
	loc:@W_n1*
_output_shapes

:

?
W_n1/Adam_1/Initializer/zerosConst*
valueB
*    *
_class
	loc:@W_n1*
dtype0*
_output_shapes

:

?
W_n1/Adam_1
VariableV2*
dtype0*
_output_shapes

:
*
shared_name *
_class
	loc:@W_n1*
	container *
shape
:

?
W_n1/Adam_1/AssignAssignW_n1/Adam_1W_n1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@W_n1*
validate_shape(*
_output_shapes

:

k
W_n1/Adam_1/readIdentityW_n1/Adam_1*
T0*
_class
	loc:@W_n1*
_output_shapes

:

?
b_n1/Adam/Initializer/zerosConst*
valueB
*    *
_class
	loc:@b_n1*
dtype0*
_output_shapes
:

?
	b_n1/Adam
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *
_class
	loc:@b_n1*
	container *
shape:

?
b_n1/Adam/AssignAssign	b_n1/Adamb_n1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_n1*
validate_shape(*
_output_shapes
:

c
b_n1/Adam/readIdentity	b_n1/Adam*
T0*
_class
	loc:@b_n1*
_output_shapes
:

?
b_n1/Adam_1/Initializer/zerosConst*
valueB
*    *
_class
	loc:@b_n1*
dtype0*
_output_shapes
:

?
b_n1/Adam_1
VariableV2*
shared_name *
_class
	loc:@b_n1*
	container *
shape:
*
dtype0*
_output_shapes
:

?
b_n1/Adam_1/AssignAssignb_n1/Adam_1b_n1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_n1*
validate_shape(*
_output_shapes
:

g
b_n1/Adam_1/readIdentityb_n1/Adam_1*
T0*
_class
	loc:@b_n1*
_output_shapes
:

W
Adam/learning_rateConst*
valueB
 *
?#<*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w??*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w?+2
?
Adam/update_W_c1/ApplyAdam	ApplyAdamW_c1	W_c1/AdamW_c1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@W_c1*
use_nesterov( *&
_output_shapes
:
?
Adam/update_b_c1/ApplyAdam	ApplyAdamb_c1	b_c1/Adamb_c1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*
_class
	loc:@b_c1
?
Adam/update_W_n1/ApplyAdam	ApplyAdamW_n1	W_n1/AdamW_n1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@W_n1*
use_nesterov( *
_output_shapes

:

?
Adam/update_b_n1/ApplyAdam	ApplyAdamb_n1	b_n1/Adamb_n1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
_class
	loc:@b_n1*
use_nesterov( *
_output_shapes
:
*
use_locking( 
?
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_W_c1/ApplyAdam^Adam/update_W_n1/ApplyAdam^Adam/update_b_c1/ApplyAdam^Adam/update_b_n1/ApplyAdam*
_output_shapes
: *
T0*
_class
	loc:@W_c1
?
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_class
	loc:@W_c1*
validate_shape(*
_output_shapes
: *
use_locking( 
?

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_W_c1/ApplyAdam^Adam/update_W_n1/ApplyAdam^Adam/update_b_c1/ApplyAdam^Adam/update_b_n1/ApplyAdam*
T0*
_class
	loc:@W_c1*
_output_shapes
: 
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
	loc:@W_c1*
validate_shape(*
_output_shapes
: 
?
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_W_c1/ApplyAdam^Adam/update_W_n1/ApplyAdam^Adam/update_b_c1/ApplyAdam^Adam/update_b_n1/ApplyAdam
?
initNoOp^W_c1/Adam/Assign^W_c1/Adam_1/Assign^W_c1/Assign^W_n1/Adam/Assign^W_n1/Adam_1/Assign^W_n1/Assign^b_c1/Adam/Assign^b_c1/Adam_1/Assign^b_c1/Assign^b_n1/Adam/Assign^b_n1/Adam_1/Assign^b_n1/Assign^beta1_power/Assign^beta2_power/Assign
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
ArgMaxArgMaxSoftmaxArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:?????????*

Tidx0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
?
ArgMax_1ArgMaxPlaceholderArgMax_1/dimension*
output_type0	*#
_output_shapes
:?????????*

Tidx0*
T0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:?????????
`
CastCastEqual*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:?????????
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Y
MeanMeanCastConst_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
?
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_9c704e8189794a2caf5116ec7dd6d334/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*?
value?B?BW_c1B	W_c1/AdamBW_c1/Adam_1BW_n1B	W_n1/AdamBW_n1/Adam_1Bb_c1B	b_c1/AdamBb_c1/Adam_1Bb_n1B	b_n1/AdamBb_n1/Adam_1Bbeta1_powerBbeta2_power

save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*/
value&B$B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesW_c1	W_c1/AdamW_c1/Adam_1W_n1	W_n1/AdamW_n1/Adam_1b_c1	b_c1/Adamb_c1/Adam_1b_n1	b_n1/Adamb_n1/Adam_1beta1_powerbeta2_power*
dtypes
2
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
?
save/RestoreV2/tensor_namesConst*?
value?B?BW_c1B	W_c1/AdamBW_c1/Adam_1BW_n1B	W_n1/AdamBW_n1/Adam_1Bb_c1B	b_c1/AdamBb_c1/Adam_1Bb_n1B	b_n1/AdamBb_n1/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:
?
save/RestoreV2/shape_and_slicesConst*/
value&B$B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*L
_output_shapes:
8::::::::::::::
?
save/AssignAssignW_c1save/RestoreV2*
use_locking(*
T0*
_class
	loc:@W_c1*
validate_shape(*&
_output_shapes
:
?
save/Assign_1Assign	W_c1/Adamsave/RestoreV2:1*
T0*
_class
	loc:@W_c1*
validate_shape(*&
_output_shapes
:*
use_locking(
?
save/Assign_2AssignW_c1/Adam_1save/RestoreV2:2*
use_locking(*
T0*
_class
	loc:@W_c1*
validate_shape(*&
_output_shapes
:
?
save/Assign_3AssignW_n1save/RestoreV2:3*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*
_class
	loc:@W_n1
?
save/Assign_4Assign	W_n1/Adamsave/RestoreV2:4*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*
_class
	loc:@W_n1
?
save/Assign_5AssignW_n1/Adam_1save/RestoreV2:5*
T0*
_class
	loc:@W_n1*
validate_shape(*
_output_shapes

:
*
use_locking(
?
save/Assign_6Assignb_c1save/RestoreV2:6*
T0*
_class
	loc:@b_c1*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_7Assign	b_c1/Adamsave/RestoreV2:7*
T0*
_class
	loc:@b_c1*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_8Assignb_c1/Adam_1save/RestoreV2:8*
use_locking(*
T0*
_class
	loc:@b_c1*
validate_shape(*
_output_shapes
:
?
save/Assign_9Assignb_n1save/RestoreV2:9*
T0*
_class
	loc:@b_n1*
validate_shape(*
_output_shapes
:
*
use_locking(
?
save/Assign_10Assign	b_n1/Adamsave/RestoreV2:10*
use_locking(*
T0*
_class
	loc:@b_n1*
validate_shape(*
_output_shapes
:

?
save/Assign_11Assignb_n1/Adam_1save/RestoreV2:11*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
	loc:@b_n1
?
save/Assign_12Assignbeta1_powersave/RestoreV2:12*
T0*
_class
	loc:@W_c1*
validate_shape(*
_output_shapes
: *
use_locking(
?
save/Assign_13Assignbeta2_powersave/RestoreV2:13*
use_locking(*
T0*
_class
	loc:@W_c1*
validate_shape(*
_output_shapes
: 
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"?
trainable_variables??
8
W_c1:0W_c1/AssignW_c1/read:02truncated_normal:08
:
b_c1:0b_c1/Assignb_c1/read:02truncated_normal_1:08
:
W_n1:0W_n1/AssignW_n1/read:02truncated_normal_2:08
:
b_n1:0b_n1/Assignb_n1/read:02truncated_normal_3:08"
train_op

Adam"?
	variables??
8
W_c1:0W_c1/AssignW_c1/read:02truncated_normal:08
:
b_c1:0b_c1/Assignb_c1/read:02truncated_normal_1:08
:
W_n1:0W_n1/AssignW_n1/read:02truncated_normal_2:08
:
b_n1:0b_n1/Assignb_n1/read:02truncated_normal_3:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
P
W_c1/Adam:0W_c1/Adam/AssignW_c1/Adam/read:02W_c1/Adam/Initializer/zeros:0
X
W_c1/Adam_1:0W_c1/Adam_1/AssignW_c1/Adam_1/read:02W_c1/Adam_1/Initializer/zeros:0
P
b_c1/Adam:0b_c1/Adam/Assignb_c1/Adam/read:02b_c1/Adam/Initializer/zeros:0
X
b_c1/Adam_1:0b_c1/Adam_1/Assignb_c1/Adam_1/read:02b_c1/Adam_1/Initializer/zeros:0
P
W_n1/Adam:0W_n1/Adam/AssignW_n1/Adam/read:02W_n1/Adam/Initializer/zeros:0
X
W_n1/Adam_1:0W_n1/Adam_1/AssignW_n1/Adam_1/read:02W_n1/Adam_1/Initializer/zeros:0
P
b_n1/Adam:0b_n1/Adam/Assignb_n1/Adam/read:02b_n1/Adam/Initializer/zeros:0
X
b_n1/Adam_1:0b_n1/Adam_1/Assignb_n1/Adam_1/read:02b_n1/Adam_1/Initializer/zeros:0*?
serving_default?
 
X
X:0??????????)
logits
logits:0?????????

wn1
W_n1:0e
#
wc1
W_c1:0e
bn1
b_n1:0e

bc1
b_c1:0etensorflow/serving/predict