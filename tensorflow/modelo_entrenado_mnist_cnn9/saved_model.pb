??
?(?(
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
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
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
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
shared_namestring ?"serve*1.11.02v1.11.0-rc2-4-gc19e29306c??
f
XPlaceholder*
dtype0*(
_output_shapes
:??????????*
shape:??????????
f
Reshape/shapeConst*%
valueB"????         *
dtype0*
_output_shapes
:
l
ReshapeReshapeXReshape/shape*
T0*
Tshape0*/
_output_shapes
:?????????
o
truncated_normal/shapeConst*%
valueB"            *
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
:*

seed 
?
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*&
_output_shapes
:*
T0
u
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*&
_output_shapes
:
?
W_c1
VariableV2*
shape:*
shared_name *
dtype0*
	container *&
_output_shapes
:
?
W_c1/AssignAssignW_c1truncated_normal*
T0*
_class
	loc:@W_c1*
validate_shape(*&
_output_shapes
:*
use_locking(
e
	W_c1/readIdentityW_c1*
T0*
_class
	loc:@W_c1*&
_output_shapes
:
b
truncated_normal_1/shapeConst*
valueB:*
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

seed *
T0*
dtype0*
seed2 *
_output_shapes
:
?
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0*
_output_shapes
:
o
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
_output_shapes
:*
T0
p
b_c1
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*
shape:
?
b_c1/AssignAssignb_c1truncated_normal_1*
use_locking(*
T0*
_class
	loc:@b_c1*
validate_shape(*
_output_shapes
:
Y
	b_c1/readIdentityb_c1*
_output_shapes
:*
T0*
_class
	loc:@b_c1
q
truncated_normal_2/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_2/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_2/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
T0*
dtype0*
seed2 *&
_output_shapes
:*

seed 
?
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0*&
_output_shapes
:
{
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0*&
_output_shapes
:
?
W_c2
VariableV2*
dtype0*
	container *&
_output_shapes
:*
shape:*
shared_name 
?
W_c2/AssignAssignW_c2truncated_normal_2*
T0*
_class
	loc:@W_c2*
validate_shape(*&
_output_shapes
:*
use_locking(
e
	W_c2/readIdentityW_c2*
T0*
_class
	loc:@W_c2*&
_output_shapes
:
b
truncated_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB:
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

seed *
T0*
dtype0*
seed2 *
_output_shapes
:
?
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0*
_output_shapes
:
o
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0*
_output_shapes
:
p
b_c2
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
?
b_c2/AssignAssignb_c2truncated_normal_3*
T0*
_class
	loc:@b_c2*
validate_shape(*
_output_shapes
:*
use_locking(
Y
	b_c2/readIdentityb_c2*
T0*
_class
	loc:@b_c2*
_output_shapes
:
q
truncated_normal_4/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_4/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_4/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*

seed *
T0*
dtype0*
seed2 *&
_output_shapes
:
?
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0*&
_output_shapes
:
{
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0*&
_output_shapes
:
?
W_c3
VariableV2*
shared_name *
dtype0*
	container *&
_output_shapes
:*
shape:
?
W_c3/AssignAssignW_c3truncated_normal_4*
use_locking(*
T0*
_class
	loc:@W_c3*
validate_shape(*&
_output_shapes
:
e
	W_c3/readIdentityW_c3*
T0*
_class
	loc:@W_c3*&
_output_shapes
:
b
truncated_normal_5/shapeConst*
valueB:*
dtype0*
_output_shapes
:
\
truncated_normal_5/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_5/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
dtype0*
seed2 *
_output_shapes
:*

seed *
T0
?
truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
T0*
_output_shapes
:
o
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
T0*
_output_shapes
:
p
b_c3
VariableV2*
shape:*
shared_name *
dtype0*
	container *
_output_shapes
:
?
b_c3/AssignAssignb_c3truncated_normal_5*
use_locking(*
T0*
_class
	loc:@b_c3*
validate_shape(*
_output_shapes
:
Y
	b_c3/readIdentityb_c3*
T0*
_class
	loc:@b_c3*
_output_shapes
:
q
truncated_normal_6/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            
\
truncated_normal_6/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_6/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_6/TruncatedNormalTruncatedNormaltruncated_normal_6/shape*
T0*
dtype0*
seed2 *&
_output_shapes
:*

seed 
?
truncated_normal_6/mulMul"truncated_normal_6/TruncatedNormaltruncated_normal_6/stddev*
T0*&
_output_shapes
:
{
truncated_normal_6Addtruncated_normal_6/multruncated_normal_6/mean*
T0*&
_output_shapes
:
?
W_c4
VariableV2*
dtype0*
	container *&
_output_shapes
:*
shape:*
shared_name 
?
W_c4/AssignAssignW_c4truncated_normal_6*
T0*
_class
	loc:@W_c4*
validate_shape(*&
_output_shapes
:*
use_locking(
e
	W_c4/readIdentityW_c4*
T0*
_class
	loc:@W_c4*&
_output_shapes
:
b
truncated_normal_7/shapeConst*
dtype0*
_output_shapes
:*
valueB:
\
truncated_normal_7/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
^
truncated_normal_7/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
"truncated_normal_7/TruncatedNormalTruncatedNormaltruncated_normal_7/shape*

seed *
T0*
dtype0*
seed2 *
_output_shapes
:
?
truncated_normal_7/mulMul"truncated_normal_7/TruncatedNormaltruncated_normal_7/stddev*
T0*
_output_shapes
:
o
truncated_normal_7Addtruncated_normal_7/multruncated_normal_7/mean*
_output_shapes
:*
T0
p
b_c4
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
?
b_c4/AssignAssignb_c4truncated_normal_7*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@b_c4
Y
	b_c4/readIdentityb_c4*
T0*
_class
	loc:@b_c4*
_output_shapes
:
q
truncated_normal_8/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
\
truncated_normal_8/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_8/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *???=
?
"truncated_normal_8/TruncatedNormalTruncatedNormaltruncated_normal_8/shape*
T0*
dtype0*
seed2 *&
_output_shapes
:*

seed 
?
truncated_normal_8/mulMul"truncated_normal_8/TruncatedNormaltruncated_normal_8/stddev*
T0*&
_output_shapes
:
{
truncated_normal_8Addtruncated_normal_8/multruncated_normal_8/mean*
T0*&
_output_shapes
:
?
W_c5
VariableV2*
shape:*
shared_name *
dtype0*
	container *&
_output_shapes
:
?
W_c5/AssignAssignW_c5truncated_normal_8*
T0*
_class
	loc:@W_c5*
validate_shape(*&
_output_shapes
:*
use_locking(
e
	W_c5/readIdentityW_c5*
T0*
_class
	loc:@W_c5*&
_output_shapes
:
b
truncated_normal_9/shapeConst*
dtype0*
_output_shapes
:*
valueB:
\
truncated_normal_9/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
truncated_normal_9/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *???=
?
"truncated_normal_9/TruncatedNormalTruncatedNormaltruncated_normal_9/shape*
T0*
dtype0*
seed2 *
_output_shapes
:*

seed 
?
truncated_normal_9/mulMul"truncated_normal_9/TruncatedNormaltruncated_normal_9/stddev*
T0*
_output_shapes
:
o
truncated_normal_9Addtruncated_normal_9/multruncated_normal_9/mean*
T0*
_output_shapes
:
p
b_c5
VariableV2*
dtype0*
	container *
_output_shapes
:*
shape:*
shared_name 
?
b_c5/AssignAssignb_c5truncated_normal_9*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@b_c5
Y
	b_c5/readIdentityb_c5*
_output_shapes
:*
T0*
_class
	loc:@b_c5
j
truncated_normal_10/shapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
]
truncated_normal_10/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
truncated_normal_10/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
#truncated_normal_10/TruncatedNormalTruncatedNormaltruncated_normal_10/shape*
T0*
dtype0*
seed2 *
_output_shapes

:
*

seed 
?
truncated_normal_10/mulMul#truncated_normal_10/TruncatedNormaltruncated_normal_10/stddev*
T0*
_output_shapes

:

v
truncated_normal_10Addtruncated_normal_10/multruncated_normal_10/mean*
T0*
_output_shapes

:

x
W_n1
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:
*
shape
:

?
W_n1/AssignAssignW_n1truncated_normal_10*
use_locking(*
T0*
_class
	loc:@W_n1*
validate_shape(*
_output_shapes

:

]
	W_n1/readIdentityW_n1*
T0*
_class
	loc:@W_n1*
_output_shapes

:

c
truncated_normal_11/shapeConst*
valueB:
*
dtype0*
_output_shapes
:
]
truncated_normal_11/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
truncated_normal_11/stddevConst*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
#truncated_normal_11/TruncatedNormalTruncatedNormaltruncated_normal_11/shape*
T0*
dtype0*
seed2 *
_output_shapes
:
*

seed 
?
truncated_normal_11/mulMul#truncated_normal_11/TruncatedNormaltruncated_normal_11/stddev*
T0*
_output_shapes
:

r
truncated_normal_11Addtruncated_normal_11/multruncated_normal_11/mean*
T0*
_output_shapes
:

p
b_n1
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:
*
shape:

?
b_n1/AssignAssignb_n1truncated_normal_11*
T0*
_class
	loc:@b_n1*
validate_shape(*
_output_shapes
:
*
use_locking(
Y
	b_n1/readIdentityb_n1*
_output_shapes
:
*
T0*
_class
	loc:@b_n1
?
Conv2DConv2DReshape	W_c1/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:?????????
S
conv1IdentityConv2D*
T0*/
_output_shapes
:?????????
u
BiasAddBiasAddconv1	b_c1/read*
data_formatNHWC*/
_output_shapes
:?????????*
T0
]
conv1_con_biasIdentityBiasAdd*/
_output_shapes
:?????????*
T0
V
ReluReluconv1_con_bias*/
_output_shapes
:?????????*
T0
`
conv1_con_activacionIdentityRelu*
T0*/
_output_shapes
:?????????
?
MaxPoolMaxPoolconv1_con_activacion*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:?????????
X
	maxPool_1IdentityMaxPool*/
_output_shapes
:?????????*
T0
?
Conv2D_1Conv2D	maxPool_1	W_c2/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:?????????
U
conv2IdentityConv2D_1*
T0*/
_output_shapes
:?????????
w
	BiasAdd_1BiasAddconv2	b_c2/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????
_
conv2_con_biasIdentity	BiasAdd_1*
T0*/
_output_shapes
:?????????
X
Relu_1Reluconv2_con_bias*
T0*/
_output_shapes
:?????????
b
conv2_con_activacionIdentityRelu_1*
T0*/
_output_shapes
:?????????
?
Conv2D_2Conv2D	maxPool_1	W_c3/read*/
_output_shapes
:?????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
U
conv3IdentityConv2D_2*
T0*/
_output_shapes
:?????????
w
	BiasAdd_2BiasAddconv3	b_c3/read*
data_formatNHWC*/
_output_shapes
:?????????*
T0
_
conv3_con_biasIdentity	BiasAdd_2*
T0*/
_output_shapes
:?????????
X
Relu_2Reluconv3_con_bias*
T0*/
_output_shapes
:?????????
b
conv3_con_activacionIdentityRelu_2*
T0*/
_output_shapes
:?????????
?
Conv2D_3Conv2Dconv2_con_activacion	W_c4/read*
paddingSAME*/
_output_shapes
:?????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
U
conv4IdentityConv2D_3*
T0*/
_output_shapes
:?????????
w
	BiasAdd_3BiasAddconv4	b_c4/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????
_
conv4_con_biasIdentity	BiasAdd_3*
T0*/
_output_shapes
:?????????
X
Relu_3Reluconv4_con_bias*
T0*/
_output_shapes
:?????????
b
conv4_con_activacionIdentityRelu_3*
T0*/
_output_shapes
:?????????
?
Conv2D_4Conv2Dconv3_con_activacion	W_c5/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:?????????*
	dilations

U
conv5IdentityConv2D_4*
T0*/
_output_shapes
:?????????
w
	BiasAdd_4BiasAddconv5	b_c5/read*
T0*
data_formatNHWC*/
_output_shapes
:?????????
_
conv5_con_biasIdentity	BiasAdd_4*
T0*/
_output_shapes
:?????????
X
Relu_4Reluconv5_con_bias*
T0*/
_output_shapes
:?????????
b
conv5_con_activacionIdentityRelu_4*/
_output_shapes
:?????????*
T0
p
addAddconv4_con_activacionconv5_con_activacion*
T0*/
_output_shapes
:?????????
O
sumaIdentityadd*
T0*/
_output_shapes
:?????????
?
AvgPoolAvgPoolsuma*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*/
_output_shapes
:?????????
^
avg_global_poolIdentityAvgPool*
T0*/
_output_shapes
:?????????
`
Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
v
	Reshape_1Reshapeavg_global_poolReshape_1/shape*
T0*
Tshape0*'
_output_shapes
:?????????
[
avg_global_pool_1DIdentity	Reshape_1*
T0*'
_output_shapes
:?????????
?
MatMulMatMulavg_global_pool_1D	W_n1/read*
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

N
add_1Addnn1	b_n1/read*
T0*'
_output_shapes
:?????????

K
logitsIdentityadd_1*'
_output_shapes
:?????????
*
T0
L
SoftmaxSoftmaxlogits*
T0*'
_output_shapes
:?????????

n
PlaceholderPlaceholder*
shape:?????????
*
dtype0*'
_output_shapes
:?????????

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
dtype0*
_output_shapes
:*
valueB"       
T
SumSummulConst*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
0
NegNegSum*
_output_shapes
: *
T0
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
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
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
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*'
_output_shapes
:?????????
*

Tmultiples0*
T0
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
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
m
gradients/mul_grad/MulMulgradients/Sum_grad/TileLog*
T0*'
_output_shapes
:?????????

?
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
?
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

w
gradients/mul_grad/Mul_1MulPlaceholdergradients/Sum_grad/Tile*'
_output_shapes
:?????????
*
T0
?
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*'
_output_shapes
:?????????
*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
?
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*'
_output_shapes
:?????????

?
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*'
_output_shapes
:?????????
*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1
?
gradients/Log_grad/Reciprocal
ReciprocalSoftmax.^gradients/mul_grad/tuple/control_dependency_1*'
_output_shapes
:?????????
*
T0
?
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*'
_output_shapes
:?????????
*
T0
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
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*
T0*'
_output_shapes
:?????????*

Tidx0*
	keep_dims(
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
]
gradients/add_1_grad/ShapeShapenn1*
_output_shapes
:*
T0*
out_type0
f
gradients/add_1_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
?
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/add_1_grad/SumSumgradients/Softmax_grad/mul_1*gradients/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*'
_output_shapes
:?????????
*
T0*
Tshape0
?
gradients/add_1_grad/Sum_1Sumgradients/Softmax_grad/mul_1,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_output_shapes
:
*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
?
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*'
_output_shapes
:?????????

?
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*
_output_shapes
:

?
gradients/MatMul_grad/MatMulMatMul-gradients/add_1_grad/tuple/control_dependency	W_n1/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:?????????
?
gradients/MatMul_grad/MatMul_1MatMulavg_global_pool_1D-gradients/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes

:

n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
?
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:?????????
?
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:

m
gradients/Reshape_1_grad/ShapeShapeavg_global_pool*
T0*
out_type0*
_output_shapes
:
?
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0*/
_output_shapes
:?????????
`
gradients/AvgPool_grad/ShapeShapesuma*
T0*
out_type0*
_output_shapes
:
?
"gradients/AvgPool_grad/AvgPoolGradAvgPoolGradgradients/AvgPool_grad/Shape gradients/Reshape_1_grad/Reshape*/
_output_shapes
:?????????*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
l
gradients/add_grad/ShapeShapeconv4_con_activacion*
T0*
out_type0*
_output_shapes
:
n
gradients/add_grad/Shape_1Shapeconv5_con_activacion*
T0*
out_type0*
_output_shapes
:
?
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/add_grad/SumSum"gradients/AvgPool_grad/AvgPoolGrad(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
?
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:?????????
?
gradients/add_grad/Sum_1Sum"gradients/AvgPool_grad/AvgPoolGrad*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*/
_output_shapes
:?????????*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
?
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*/
_output_shapes
:?????????
?
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_output_shapes
:?????????*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
?
gradients/Relu_3_grad/ReluGradReluGrad+gradients/add_grad/tuple/control_dependencyRelu_3*
T0*/
_output_shapes
:?????????
?
gradients/Relu_4_grad/ReluGradReluGrad-gradients/add_grad/tuple/control_dependency_1Relu_4*/
_output_shapes
:?????????*
T0
?
$gradients/BiasAdd_3_grad/BiasAddGradBiasAddGradgradients/Relu_3_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
y
)gradients/BiasAdd_3_grad/tuple/group_depsNoOp%^gradients/BiasAdd_3_grad/BiasAddGrad^gradients/Relu_3_grad/ReluGrad
?
1gradients/BiasAdd_3_grad/tuple/control_dependencyIdentitygradients/Relu_3_grad/ReluGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*/
_output_shapes
:?????????*
T0*1
_class'
%#loc:@gradients/Relu_3_grad/ReluGrad
?
3gradients/BiasAdd_3_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_3_grad/BiasAddGrad*^gradients/BiasAdd_3_grad/tuple/group_deps*
_output_shapes
:*
T0*7
_class-
+)loc:@gradients/BiasAdd_3_grad/BiasAddGrad
?
$gradients/BiasAdd_4_grad/BiasAddGradBiasAddGradgradients/Relu_4_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
y
)gradients/BiasAdd_4_grad/tuple/group_depsNoOp%^gradients/BiasAdd_4_grad/BiasAddGrad^gradients/Relu_4_grad/ReluGrad
?
1gradients/BiasAdd_4_grad/tuple/control_dependencyIdentitygradients/Relu_4_grad/ReluGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_4_grad/ReluGrad*/
_output_shapes
:?????????
?
3gradients/BiasAdd_4_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_4_grad/BiasAddGrad*^gradients/BiasAdd_4_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_4_grad/BiasAddGrad*
_output_shapes
:
?
gradients/Conv2D_3_grad/ShapeNShapeNconv2_con_activacion	W_c4/read*
T0*
out_type0*
N* 
_output_shapes
::
?
+gradients/Conv2D_3_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_3_grad/ShapeN	W_c4/read1gradients/BiasAdd_3_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:?????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
,gradients/Conv2D_3_grad/Conv2DBackpropFilterConv2DBackpropFilterconv2_con_activacion gradients/Conv2D_3_grad/ShapeN:11gradients/BiasAdd_3_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0
?
(gradients/Conv2D_3_grad/tuple/group_depsNoOp-^gradients/Conv2D_3_grad/Conv2DBackpropFilter,^gradients/Conv2D_3_grad/Conv2DBackpropInput
?
0gradients/Conv2D_3_grad/tuple/control_dependencyIdentity+gradients/Conv2D_3_grad/Conv2DBackpropInput)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_3_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
2gradients/Conv2D_3_grad/tuple/control_dependency_1Identity,gradients/Conv2D_3_grad/Conv2DBackpropFilter)^gradients/Conv2D_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_3_grad/Conv2DBackpropFilter*&
_output_shapes
:
?
gradients/Conv2D_4_grad/ShapeNShapeNconv3_con_activacion	W_c5/read*
T0*
out_type0*
N* 
_output_shapes
::
?
+gradients/Conv2D_4_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_4_grad/ShapeN	W_c5/read1gradients/BiasAdd_4_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:?????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
,gradients/Conv2D_4_grad/Conv2DBackpropFilterConv2DBackpropFilterconv3_con_activacion gradients/Conv2D_4_grad/ShapeN:11gradients/BiasAdd_4_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0
?
(gradients/Conv2D_4_grad/tuple/group_depsNoOp-^gradients/Conv2D_4_grad/Conv2DBackpropFilter,^gradients/Conv2D_4_grad/Conv2DBackpropInput
?
0gradients/Conv2D_4_grad/tuple/control_dependencyIdentity+gradients/Conv2D_4_grad/Conv2DBackpropInput)^gradients/Conv2D_4_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_4_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
2gradients/Conv2D_4_grad/tuple/control_dependency_1Identity,gradients/Conv2D_4_grad/Conv2DBackpropFilter)^gradients/Conv2D_4_grad/tuple/group_deps*&
_output_shapes
:*
T0*?
_class5
31loc:@gradients/Conv2D_4_grad/Conv2DBackpropFilter
?
gradients/Relu_1_grad/ReluGradReluGrad0gradients/Conv2D_3_grad/tuple/control_dependencyRelu_1*
T0*/
_output_shapes
:?????????
?
gradients/Relu_2_grad/ReluGradReluGrad0gradients/Conv2D_4_grad/tuple/control_dependencyRelu_2*
T0*/
_output_shapes
:?????????
?
$gradients/BiasAdd_1_grad/BiasAddGradBiasAddGradgradients/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
y
)gradients/BiasAdd_1_grad/tuple/group_depsNoOp%^gradients/BiasAdd_1_grad/BiasAddGrad^gradients/Relu_1_grad/ReluGrad
?
1gradients/BiasAdd_1_grad/tuple/control_dependencyIdentitygradients/Relu_1_grad/ReluGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*/
_output_shapes
:?????????*
T0*1
_class'
%#loc:@gradients/Relu_1_grad/ReluGrad
?
3gradients/BiasAdd_1_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_1_grad/BiasAddGrad*^gradients/BiasAdd_1_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/BiasAdd_1_grad/BiasAddGrad*
_output_shapes
:
?
$gradients/BiasAdd_2_grad/BiasAddGradBiasAddGradgradients/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
y
)gradients/BiasAdd_2_grad/tuple/group_depsNoOp%^gradients/BiasAdd_2_grad/BiasAddGrad^gradients/Relu_2_grad/ReluGrad
?
1gradients/BiasAdd_2_grad/tuple/control_dependencyIdentitygradients/Relu_2_grad/ReluGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Relu_2_grad/ReluGrad*/
_output_shapes
:?????????
?
3gradients/BiasAdd_2_grad/tuple/control_dependency_1Identity$gradients/BiasAdd_2_grad/BiasAddGrad*^gradients/BiasAdd_2_grad/tuple/group_deps*
_output_shapes
:*
T0*7
_class-
+)loc:@gradients/BiasAdd_2_grad/BiasAddGrad
?
gradients/Conv2D_1_grad/ShapeNShapeN	maxPool_1	W_c2/read*
T0*
out_type0*
N* 
_output_shapes
::
?
+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeN	W_c2/read1gradients/BiasAdd_1_grad/tuple/control_dependency*/
_output_shapes
:?????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
?
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilter	maxPool_1 gradients/Conv2D_1_grad/ShapeN:11gradients/BiasAdd_1_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0
?
(gradients/Conv2D_1_grad/tuple/group_depsNoOp-^gradients/Conv2D_1_grad/Conv2DBackpropFilter,^gradients/Conv2D_1_grad/Conv2DBackpropInput
?
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*&
_output_shapes
:*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
?
gradients/Conv2D_2_grad/ShapeNShapeN	maxPool_1	W_c3/read*
T0*
out_type0*
N* 
_output_shapes
::
?
+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeN	W_c3/read1gradients/BiasAdd_2_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:?????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilter	maxPool_1 gradients/Conv2D_2_grad/ShapeN:11gradients/BiasAdd_2_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
(gradients/Conv2D_2_grad/tuple/group_depsNoOp-^gradients/Conv2D_2_grad/Conv2DBackpropFilter,^gradients/Conv2D_2_grad/Conv2DBackpropInput
?
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput*/
_output_shapes
:?????????
?
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter*&
_output_shapes
:
?
gradients/AddNAddN0gradients/Conv2D_1_grad/tuple/control_dependency0gradients/Conv2D_2_grad/tuple/control_dependency*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput*
N*/
_output_shapes
:?????????
?
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradconv1_con_activacionMaxPoolgradients/AddN*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:?????????
?
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0*/
_output_shapes
:?????????
?
"gradients/BiasAdd_grad/BiasAddGradBiasAddGradgradients/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:*
T0
s
'gradients/BiasAdd_grad/tuple/group_depsNoOp#^gradients/BiasAdd_grad/BiasAddGrad^gradients/Relu_grad/ReluGrad
?
/gradients/BiasAdd_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*/
_output_shapes
:?????????
?
1gradients/BiasAdd_grad/tuple/control_dependency_1Identity"gradients/BiasAdd_grad/BiasAddGrad(^gradients/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
~
gradients/Conv2D_grad/ShapeNShapeNReshape	W_c1/read*
N* 
_output_shapes
::*
T0*
out_type0
?
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeN	W_c1/read/gradients/BiasAdd_grad/tuple/control_dependency*
paddingSAME*/
_output_shapes
:?????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterReshapegradients/Conv2D_grad/ShapeN:1/gradients/BiasAdd_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
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
:
w
beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*
_class
	loc:@W_c1
?
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
	loc:@W_c1*
	container *
shape: 
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
VariableV2*
_class
	loc:@W_c1*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
	loc:@W_c1
c
beta2_power/readIdentitybeta2_power*
T0*
_class
	loc:@W_c1*
_output_shapes
: 
?
W_c1/Adam/Initializer/zerosConst*
dtype0*&
_output_shapes
:*%
valueB*    *
_class
	loc:@W_c1
?
	W_c1/Adam
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *
_class
	loc:@W_c1*
	container *
shape:
?
W_c1/Adam/AssignAssign	W_c1/AdamW_c1/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@W_c1
o
W_c1/Adam/readIdentity	W_c1/Adam*&
_output_shapes
:*
T0*
_class
	loc:@W_c1
?
W_c1/Adam_1/Initializer/zerosConst*%
valueB*    *
_class
	loc:@W_c1*
dtype0*&
_output_shapes
:
?
W_c1/Adam_1
VariableV2*
shared_name *
_class
	loc:@W_c1*
	container *
shape:*
dtype0*&
_output_shapes
:
?
W_c1/Adam_1/AssignAssignW_c1/Adam_1W_c1/Adam_1/Initializer/zeros*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@W_c1
s
W_c1/Adam_1/readIdentityW_c1/Adam_1*
T0*
_class
	loc:@W_c1*&
_output_shapes
:
?
b_c1/Adam/Initializer/zerosConst*
valueB*    *
_class
	loc:@b_c1*
dtype0*
_output_shapes
:
?
	b_c1/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
	loc:@b_c1*
	container *
shape:
?
b_c1/Adam/AssignAssign	b_c1/Adamb_c1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_c1*
validate_shape(*
_output_shapes
:
c
b_c1/Adam/readIdentity	b_c1/Adam*
T0*
_class
	loc:@b_c1*
_output_shapes
:
?
b_c1/Adam_1/Initializer/zerosConst*
valueB*    *
_class
	loc:@b_c1*
dtype0*
_output_shapes
:
?
b_c1/Adam_1
VariableV2*
shared_name *
_class
	loc:@b_c1*
	container *
shape:*
dtype0*
_output_shapes
:
?
b_c1/Adam_1/AssignAssignb_c1/Adam_1b_c1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_c1*
validate_shape(*
_output_shapes
:
g
b_c1/Adam_1/readIdentityb_c1/Adam_1*
T0*
_class
	loc:@b_c1*
_output_shapes
:
?
W_c2/Adam/Initializer/zerosConst*%
valueB*    *
_class
	loc:@W_c2*
dtype0*&
_output_shapes
:
?
	W_c2/Adam
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *
_class
	loc:@W_c2*
	container 
?
W_c2/Adam/AssignAssign	W_c2/AdamW_c2/Adam/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@W_c2*
validate_shape(*&
_output_shapes
:
o
W_c2/Adam/readIdentity	W_c2/Adam*
T0*
_class
	loc:@W_c2*&
_output_shapes
:
?
W_c2/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
:*%
valueB*    *
_class
	loc:@W_c2
?
W_c2/Adam_1
VariableV2*
shared_name *
_class
	loc:@W_c2*
	container *
shape:*
dtype0*&
_output_shapes
:
?
W_c2/Adam_1/AssignAssignW_c2/Adam_1W_c2/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@W_c2*
validate_shape(*&
_output_shapes
:
s
W_c2/Adam_1/readIdentityW_c2/Adam_1*
T0*
_class
	loc:@W_c2*&
_output_shapes
:
?
b_c2/Adam/Initializer/zerosConst*
valueB*    *
_class
	loc:@b_c2*
dtype0*
_output_shapes
:
?
	b_c2/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
	loc:@b_c2*
	container 
?
b_c2/Adam/AssignAssign	b_c2/Adamb_c2/Adam/Initializer/zeros*
T0*
_class
	loc:@b_c2*
validate_shape(*
_output_shapes
:*
use_locking(
c
b_c2/Adam/readIdentity	b_c2/Adam*
T0*
_class
	loc:@b_c2*
_output_shapes
:
?
b_c2/Adam_1/Initializer/zerosConst*
valueB*    *
_class
	loc:@b_c2*
dtype0*
_output_shapes
:
?
b_c2/Adam_1
VariableV2*
_class
	loc:@b_c2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
?
b_c2/Adam_1/AssignAssignb_c2/Adam_1b_c2/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@b_c2
g
b_c2/Adam_1/readIdentityb_c2/Adam_1*
T0*
_class
	loc:@b_c2*
_output_shapes
:
?
W_c3/Adam/Initializer/zerosConst*%
valueB*    *
_class
	loc:@W_c3*
dtype0*&
_output_shapes
:
?
	W_c3/Adam
VariableV2*
shared_name *
_class
	loc:@W_c3*
	container *
shape:*
dtype0*&
_output_shapes
:
?
W_c3/Adam/AssignAssign	W_c3/AdamW_c3/Adam/Initializer/zeros*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@W_c3
o
W_c3/Adam/readIdentity	W_c3/Adam*
T0*
_class
	loc:@W_c3*&
_output_shapes
:
?
W_c3/Adam_1/Initializer/zerosConst*
dtype0*&
_output_shapes
:*%
valueB*    *
_class
	loc:@W_c3
?
W_c3/Adam_1
VariableV2*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name *
_class
	loc:@W_c3
?
W_c3/Adam_1/AssignAssignW_c3/Adam_1W_c3/Adam_1/Initializer/zeros*
T0*
_class
	loc:@W_c3*
validate_shape(*&
_output_shapes
:*
use_locking(
s
W_c3/Adam_1/readIdentityW_c3/Adam_1*
T0*
_class
	loc:@W_c3*&
_output_shapes
:
?
b_c3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
	loc:@b_c3
?
	b_c3/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
	loc:@b_c3*
	container *
shape:
?
b_c3/Adam/AssignAssign	b_c3/Adamb_c3/Adam/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_c3*
validate_shape(*
_output_shapes
:
c
b_c3/Adam/readIdentity	b_c3/Adam*
T0*
_class
	loc:@b_c3*
_output_shapes
:
?
b_c3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
	loc:@b_c3
?
b_c3/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
	loc:@b_c3*
	container *
shape:
?
b_c3/Adam_1/AssignAssignb_c3/Adam_1b_c3/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_c3*
validate_shape(*
_output_shapes
:
g
b_c3/Adam_1/readIdentityb_c3/Adam_1*
T0*
_class
	loc:@b_c3*
_output_shapes
:
?
W_c4/Adam/Initializer/zerosConst*%
valueB*    *
_class
	loc:@W_c4*
dtype0*&
_output_shapes
:
?
	W_c4/Adam
VariableV2*
_class
	loc:@W_c4*
	container *
shape:*
dtype0*&
_output_shapes
:*
shared_name 
?
W_c4/Adam/AssignAssign	W_c4/AdamW_c4/Adam/Initializer/zeros*
T0*
_class
	loc:@W_c4*
validate_shape(*&
_output_shapes
:*
use_locking(
o
W_c4/Adam/readIdentity	W_c4/Adam*
T0*
_class
	loc:@W_c4*&
_output_shapes
:
?
W_c4/Adam_1/Initializer/zerosConst*%
valueB*    *
_class
	loc:@W_c4*
dtype0*&
_output_shapes
:
?
W_c4/Adam_1
VariableV2*
shape:*
dtype0*&
_output_shapes
:*
shared_name *
_class
	loc:@W_c4*
	container 
?
W_c4/Adam_1/AssignAssignW_c4/Adam_1W_c4/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@W_c4*
validate_shape(*&
_output_shapes
:
s
W_c4/Adam_1/readIdentityW_c4/Adam_1*
T0*
_class
	loc:@W_c4*&
_output_shapes
:
?
b_c4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
	loc:@b_c4
?
	b_c4/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
	loc:@b_c4*
	container *
shape:
?
b_c4/Adam/AssignAssign	b_c4/Adamb_c4/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@b_c4
c
b_c4/Adam/readIdentity	b_c4/Adam*
T0*
_class
	loc:@b_c4*
_output_shapes
:
?
b_c4/Adam_1/Initializer/zerosConst*
valueB*    *
_class
	loc:@b_c4*
dtype0*
_output_shapes
:
?
b_c4/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
	loc:@b_c4
?
b_c4/Adam_1/AssignAssignb_c4/Adam_1b_c4/Adam_1/Initializer/zeros*
T0*
_class
	loc:@b_c4*
validate_shape(*
_output_shapes
:*
use_locking(
g
b_c4/Adam_1/readIdentityb_c4/Adam_1*
T0*
_class
	loc:@b_c4*
_output_shapes
:
?
W_c5/Adam/Initializer/zerosConst*%
valueB*    *
_class
	loc:@W_c5*
dtype0*&
_output_shapes
:
?
	W_c5/Adam
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *
_class
	loc:@W_c5*
	container *
shape:
?
W_c5/Adam/AssignAssign	W_c5/AdamW_c5/Adam/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@W_c5*
validate_shape(*&
_output_shapes
:
o
W_c5/Adam/readIdentity	W_c5/Adam*
T0*
_class
	loc:@W_c5*&
_output_shapes
:
?
W_c5/Adam_1/Initializer/zerosConst*%
valueB*    *
_class
	loc:@W_c5*
dtype0*&
_output_shapes
:
?
W_c5/Adam_1
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *
_class
	loc:@W_c5*
	container *
shape:
?
W_c5/Adam_1/AssignAssignW_c5/Adam_1W_c5/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@W_c5*
validate_shape(*&
_output_shapes
:
s
W_c5/Adam_1/readIdentityW_c5/Adam_1*
T0*
_class
	loc:@W_c5*&
_output_shapes
:
?
b_c5/Adam/Initializer/zerosConst*
valueB*    *
_class
	loc:@b_c5*
dtype0*
_output_shapes
:
?
	b_c5/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *
_class
	loc:@b_c5*
	container 
?
b_c5/Adam/AssignAssign	b_c5/Adamb_c5/Adam/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@b_c5*
validate_shape(*
_output_shapes
:
c
b_c5/Adam/readIdentity	b_c5/Adam*
T0*
_class
	loc:@b_c5*
_output_shapes
:
?
b_c5/Adam_1/Initializer/zerosConst*
valueB*    *
_class
	loc:@b_c5*
dtype0*
_output_shapes
:
?
b_c5/Adam_1
VariableV2*
shared_name *
_class
	loc:@b_c5*
	container *
shape:*
dtype0*
_output_shapes
:
?
b_c5/Adam_1/AssignAssignb_c5/Adam_1b_c5/Adam_1/Initializer/zeros*
T0*
_class
	loc:@b_c5*
validate_shape(*
_output_shapes
:*
use_locking(
g
b_c5/Adam_1/readIdentityb_c5/Adam_1*
T0*
_class
	loc:@b_c5*
_output_shapes
:
?
W_n1/Adam/Initializer/zerosConst*
valueB
*    *
_class
	loc:@W_n1*
dtype0*
_output_shapes

:

?
	W_n1/Adam
VariableV2*
shared_name *
_class
	loc:@W_n1*
	container *
shape
:
*
dtype0*
_output_shapes

:

?
W_n1/Adam/AssignAssign	W_n1/AdamW_n1/Adam/Initializer/zeros*
T0*
_class
	loc:@W_n1*
validate_shape(*
_output_shapes

:
*
use_locking(
g
W_n1/Adam/readIdentity	W_n1/Adam*
T0*
_class
	loc:@W_n1*
_output_shapes

:

?
W_n1/Adam_1/Initializer/zerosConst*
valueB
*    *
_class
	loc:@W_n1*
dtype0*
_output_shapes

:

?
W_n1/Adam_1
VariableV2*
_class
	loc:@W_n1*
	container *
shape
:
*
dtype0*
_output_shapes

:
*
shared_name 
?
W_n1/Adam_1/AssignAssignW_n1/Adam_1W_n1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@W_n1*
validate_shape(*
_output_shapes

:

k
W_n1/Adam_1/readIdentityW_n1/Adam_1*
T0*
_class
	loc:@W_n1*
_output_shapes

:

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
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
shared_name *
_class
	loc:@b_n1*
	container 
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
VariableV2*
	container *
shape:
*
dtype0*
_output_shapes
:
*
shared_name *
_class
	loc:@b_n1
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
ף;*
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
Adam/epsilonConst*
valueB
 *w?+2*
dtype0*
_output_shapes
: 
?
Adam/update_W_c1/ApplyAdam	ApplyAdamW_c1	W_c1/AdamW_c1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
use_nesterov( *&
_output_shapes
:*
use_locking( *
T0*
_class
	loc:@W_c1
?
Adam/update_b_c1/ApplyAdam	ApplyAdamb_c1	b_c1/Adamb_c1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon1gradients/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@b_c1*
use_nesterov( *
_output_shapes
:
?
Adam/update_W_c2/ApplyAdam	ApplyAdamW_c2	W_c2/AdamW_c2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
_class
	loc:@W_c2*
use_nesterov( *&
_output_shapes
:*
use_locking( 
?
Adam/update_b_c2/ApplyAdam	ApplyAdamb_c2	b_c2/Adamb_c2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@b_c2*
use_nesterov( *
_output_shapes
:
?
Adam/update_W_c3/ApplyAdam	ApplyAdamW_c3	W_c3/AdamW_c3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@W_c3*
use_nesterov( *&
_output_shapes
:
?
Adam/update_b_c3/ApplyAdam	ApplyAdamb_c3	b_c3/Adamb_c3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_2_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*
_class
	loc:@b_c3
?
Adam/update_W_c4/ApplyAdam	ApplyAdamW_c4	W_c4/AdamW_c4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_3_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@W_c4*
use_nesterov( *&
_output_shapes
:
?
Adam/update_b_c4/ApplyAdam	ApplyAdamb_c4	b_c4/Adamb_c4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_3_grad/tuple/control_dependency_1*
T0*
_class
	loc:@b_c4*
use_nesterov( *
_output_shapes
:*
use_locking( 
?
Adam/update_W_c5/ApplyAdam	ApplyAdamW_c5	W_c5/AdamW_c5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_4_grad/tuple/control_dependency_1*
T0*
_class
	loc:@W_c5*
use_nesterov( *&
_output_shapes
:*
use_locking( 
?
Adam/update_b_c5/ApplyAdam	ApplyAdamb_c5	b_c5/Adamb_c5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/BiasAdd_4_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@b_c5*
use_nesterov( *
_output_shapes
:
?
Adam/update_W_n1/ApplyAdam	ApplyAdamW_n1	W_n1/AdamW_n1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
_class
	loc:@W_n1*
use_nesterov( *
_output_shapes

:
*
use_locking( 
?
Adam/update_b_n1/ApplyAdam	ApplyAdamb_n1	b_n1/Adamb_n1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@b_n1*
use_nesterov( *
_output_shapes
:

?
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_W_c1/ApplyAdam^Adam/update_W_c2/ApplyAdam^Adam/update_W_c3/ApplyAdam^Adam/update_W_c4/ApplyAdam^Adam/update_W_c5/ApplyAdam^Adam/update_W_n1/ApplyAdam^Adam/update_b_c1/ApplyAdam^Adam/update_b_c2/ApplyAdam^Adam/update_b_c3/ApplyAdam^Adam/update_b_c4/ApplyAdam^Adam/update_b_c5/ApplyAdam^Adam/update_b_n1/ApplyAdam*
_output_shapes
: *
T0*
_class
	loc:@W_c1
?
Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
	loc:@W_c1
?

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_W_c1/ApplyAdam^Adam/update_W_c2/ApplyAdam^Adam/update_W_c3/ApplyAdam^Adam/update_W_c4/ApplyAdam^Adam/update_W_c5/ApplyAdam^Adam/update_W_n1/ApplyAdam^Adam/update_b_c1/ApplyAdam^Adam/update_b_c2/ApplyAdam^Adam/update_b_c3/ApplyAdam^Adam/update_b_c4/ApplyAdam^Adam/update_b_c5/ApplyAdam^Adam/update_b_n1/ApplyAdam*
T0*
_class
	loc:@W_c1*
_output_shapes
: 
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
	loc:@W_c1
?
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_W_c1/ApplyAdam^Adam/update_W_c2/ApplyAdam^Adam/update_W_c3/ApplyAdam^Adam/update_W_c4/ApplyAdam^Adam/update_W_c5/ApplyAdam^Adam/update_W_n1/ApplyAdam^Adam/update_b_c1/ApplyAdam^Adam/update_b_c2/ApplyAdam^Adam/update_b_c3/ApplyAdam^Adam/update_b_c4/ApplyAdam^Adam/update_b_c5/ApplyAdam^Adam/update_b_n1/ApplyAdam
?
initNoOp^W_c1/Adam/Assign^W_c1/Adam_1/Assign^W_c1/Assign^W_c2/Adam/Assign^W_c2/Adam_1/Assign^W_c2/Assign^W_c3/Adam/Assign^W_c3/Adam_1/Assign^W_c3/Assign^W_c4/Adam/Assign^W_c4/Adam_1/Assign^W_c4/Assign^W_c5/Adam/Assign^W_c5/Adam_1/Assign^W_c5/Assign^W_n1/Adam/Assign^W_n1/Adam_1/Assign^W_n1/Assign^b_c1/Adam/Assign^b_c1/Adam_1/Assign^b_c1/Assign^b_c2/Adam/Assign^b_c2/Adam_1/Assign^b_c2/Assign^b_c3/Adam/Assign^b_c3/Adam_1/Assign^b_c3/Assign^b_c4/Adam/Assign^b_c4/Adam_1/Assign^b_c4/Assign^b_c5/Adam/Assign^b_c5/Adam_1/Assign^b_c5/Assign^b_n1/Adam/Assign^b_n1/Adam_1/Assign^b_n1/Assign^beta1_power/Assign^beta2_power/Assign
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
x
ArgMaxArgMaxSoftmaxArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:?????????
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
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
MeanMeanCastConst_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
?
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_22dfbc9d9e2144ff876339372ea57445/part*
dtype0*
_output_shapes
: 
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
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:&*?
value?B?&BW_c1B	W_c1/AdamBW_c1/Adam_1BW_c2B	W_c2/AdamBW_c2/Adam_1BW_c3B	W_c3/AdamBW_c3/Adam_1BW_c4B	W_c4/AdamBW_c4/Adam_1BW_c5B	W_c5/AdamBW_c5/Adam_1BW_n1B	W_n1/AdamBW_n1/Adam_1Bb_c1B	b_c1/AdamBb_c1/Adam_1Bb_c2B	b_c2/AdamBb_c2/Adam_1Bb_c3B	b_c3/AdamBb_c3/Adam_1Bb_c4B	b_c4/AdamBb_c4/Adam_1Bb_c5B	b_c5/AdamBb_c5/Adam_1Bb_n1B	b_n1/AdamBb_n1/Adam_1Bbeta1_powerBbeta2_power
?
save/SaveV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesW_c1	W_c1/AdamW_c1/Adam_1W_c2	W_c2/AdamW_c2/Adam_1W_c3	W_c3/AdamW_c3/Adam_1W_c4	W_c4/AdamW_c4/Adam_1W_c5	W_c5/AdamW_c5/Adam_1W_n1	W_n1/AdamW_n1/Adam_1b_c1	b_c1/Adamb_c1/Adam_1b_c2	b_c2/Adamb_c2/Adam_1b_c3	b_c3/Adamb_c3/Adam_1b_c4	b_c4/Adamb_c4/Adam_1b_c5	b_c5/Adamb_c5/Adam_1b_n1	b_n1/Adamb_n1/Adam_1beta1_powerbeta2_power*4
dtypes*
(2&
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
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
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:&*?
value?B?&BW_c1B	W_c1/AdamBW_c1/Adam_1BW_c2B	W_c2/AdamBW_c2/Adam_1BW_c3B	W_c3/AdamBW_c3/Adam_1BW_c4B	W_c4/AdamBW_c4/Adam_1BW_c5B	W_c5/AdamBW_c5/Adam_1BW_n1B	W_n1/AdamBW_n1/Adam_1Bb_c1B	b_c1/AdamBb_c1/Adam_1Bb_c2B	b_c2/AdamBb_c2/Adam_1Bb_c3B	b_c3/AdamBb_c3/Adam_1Bb_c4B	b_c4/AdamBb_c4/Adam_1Bb_c5B	b_c5/AdamBb_c5/Adam_1Bb_n1B	b_n1/AdamBb_n1/Adam_1Bbeta1_powerBbeta2_power
?
save/RestoreV2/shape_and_slicesConst*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:&
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&
?
save/AssignAssignW_c1save/RestoreV2*
use_locking(*
T0*
_class
	loc:@W_c1*
validate_shape(*&
_output_shapes
:
?
save/Assign_1Assign	W_c1/Adamsave/RestoreV2:1*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@W_c1
?
save/Assign_2AssignW_c1/Adam_1save/RestoreV2:2*
T0*
_class
	loc:@W_c1*
validate_shape(*&
_output_shapes
:*
use_locking(
?
save/Assign_3AssignW_c2save/RestoreV2:3*
T0*
_class
	loc:@W_c2*
validate_shape(*&
_output_shapes
:*
use_locking(
?
save/Assign_4Assign	W_c2/Adamsave/RestoreV2:4*
use_locking(*
T0*
_class
	loc:@W_c2*
validate_shape(*&
_output_shapes
:
?
save/Assign_5AssignW_c2/Adam_1save/RestoreV2:5*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@W_c2
?
save/Assign_6AssignW_c3save/RestoreV2:6*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@W_c3
?
save/Assign_7Assign	W_c3/Adamsave/RestoreV2:7*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@W_c3
?
save/Assign_8AssignW_c3/Adam_1save/RestoreV2:8*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@W_c3
?
save/Assign_9AssignW_c4save/RestoreV2:9*
T0*
_class
	loc:@W_c4*
validate_shape(*&
_output_shapes
:*
use_locking(
?
save/Assign_10Assign	W_c4/Adamsave/RestoreV2:10*
use_locking(*
T0*
_class
	loc:@W_c4*
validate_shape(*&
_output_shapes
:
?
save/Assign_11AssignW_c4/Adam_1save/RestoreV2:11*
use_locking(*
T0*
_class
	loc:@W_c4*
validate_shape(*&
_output_shapes
:
?
save/Assign_12AssignW_c5save/RestoreV2:12*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@W_c5
?
save/Assign_13Assign	W_c5/Adamsave/RestoreV2:13*
T0*
_class
	loc:@W_c5*
validate_shape(*&
_output_shapes
:*
use_locking(
?
save/Assign_14AssignW_c5/Adam_1save/RestoreV2:14*
T0*
_class
	loc:@W_c5*
validate_shape(*&
_output_shapes
:*
use_locking(
?
save/Assign_15AssignW_n1save/RestoreV2:15*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*
_class
	loc:@W_n1
?
save/Assign_16Assign	W_n1/Adamsave/RestoreV2:16*
T0*
_class
	loc:@W_n1*
validate_shape(*
_output_shapes

:
*
use_locking(
?
save/Assign_17AssignW_n1/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes

:
*
use_locking(*
T0*
_class
	loc:@W_n1
?
save/Assign_18Assignb_c1save/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@b_c1
?
save/Assign_19Assign	b_c1/Adamsave/RestoreV2:19*
T0*
_class
	loc:@b_c1*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_20Assignb_c1/Adam_1save/RestoreV2:20*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@b_c1
?
save/Assign_21Assignb_c2save/RestoreV2:21*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@b_c2
?
save/Assign_22Assign	b_c2/Adamsave/RestoreV2:22*
T0*
_class
	loc:@b_c2*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_23Assignb_c2/Adam_1save/RestoreV2:23*
use_locking(*
T0*
_class
	loc:@b_c2*
validate_shape(*
_output_shapes
:
?
save/Assign_24Assignb_c3save/RestoreV2:24*
use_locking(*
T0*
_class
	loc:@b_c3*
validate_shape(*
_output_shapes
:
?
save/Assign_25Assign	b_c3/Adamsave/RestoreV2:25*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@b_c3
?
save/Assign_26Assignb_c3/Adam_1save/RestoreV2:26*
T0*
_class
	loc:@b_c3*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_27Assignb_c4save/RestoreV2:27*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@b_c4
?
save/Assign_28Assign	b_c4/Adamsave/RestoreV2:28*
use_locking(*
T0*
_class
	loc:@b_c4*
validate_shape(*
_output_shapes
:
?
save/Assign_29Assignb_c4/Adam_1save/RestoreV2:29*
T0*
_class
	loc:@b_c4*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_30Assignb_c5save/RestoreV2:30*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*
_class
	loc:@b_c5
?
save/Assign_31Assign	b_c5/Adamsave/RestoreV2:31*
T0*
_class
	loc:@b_c5*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_32Assignb_c5/Adam_1save/RestoreV2:32*
use_locking(*
T0*
_class
	loc:@b_c5*
validate_shape(*
_output_shapes
:
?
save/Assign_33Assignb_n1save/RestoreV2:33*
use_locking(*
T0*
_class
	loc:@b_n1*
validate_shape(*
_output_shapes
:

?
save/Assign_34Assign	b_n1/Adamsave/RestoreV2:34*
T0*
_class
	loc:@b_n1*
validate_shape(*
_output_shapes
:
*
use_locking(
?
save/Assign_35Assignb_n1/Adam_1save/RestoreV2:35*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
	loc:@b_n1
?
save/Assign_36Assignbeta1_powersave/RestoreV2:36*
use_locking(*
T0*
_class
	loc:@W_c1*
validate_shape(*
_output_shapes
: 
?
save/Assign_37Assignbeta2_powersave/RestoreV2:37*
T0*
_class
	loc:@W_c1*
validate_shape(*
_output_shapes
: *
use_locking(
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"?
trainable_variables??
8
W_c1:0W_c1/AssignW_c1/read:02truncated_normal:08
:
b_c1:0b_c1/Assignb_c1/read:02truncated_normal_1:08
:
W_c2:0W_c2/AssignW_c2/read:02truncated_normal_2:08
:
b_c2:0b_c2/Assignb_c2/read:02truncated_normal_3:08
:
W_c3:0W_c3/AssignW_c3/read:02truncated_normal_4:08
:
b_c3:0b_c3/Assignb_c3/read:02truncated_normal_5:08
:
W_c4:0W_c4/AssignW_c4/read:02truncated_normal_6:08
:
b_c4:0b_c4/Assignb_c4/read:02truncated_normal_7:08
:
W_c5:0W_c5/AssignW_c5/read:02truncated_normal_8:08
:
b_c5:0b_c5/Assignb_c5/read:02truncated_normal_9:08
;
W_n1:0W_n1/AssignW_n1/read:02truncated_normal_10:08
;
b_n1:0b_n1/Assignb_n1/read:02truncated_normal_11:08"
train_op

Adam"?
	variables??
8
W_c1:0W_c1/AssignW_c1/read:02truncated_normal:08
:
b_c1:0b_c1/Assignb_c1/read:02truncated_normal_1:08
:
W_c2:0W_c2/AssignW_c2/read:02truncated_normal_2:08
:
b_c2:0b_c2/Assignb_c2/read:02truncated_normal_3:08
:
W_c3:0W_c3/AssignW_c3/read:02truncated_normal_4:08
:
b_c3:0b_c3/Assignb_c3/read:02truncated_normal_5:08
:
W_c4:0W_c4/AssignW_c4/read:02truncated_normal_6:08
:
b_c4:0b_c4/Assignb_c4/read:02truncated_normal_7:08
:
W_c5:0W_c5/AssignW_c5/read:02truncated_normal_8:08
:
b_c5:0b_c5/Assignb_c5/read:02truncated_normal_9:08
;
W_n1:0W_n1/AssignW_n1/read:02truncated_normal_10:08
;
b_n1:0b_n1/Assignb_n1/read:02truncated_normal_11:08
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
W_c2/Adam:0W_c2/Adam/AssignW_c2/Adam/read:02W_c2/Adam/Initializer/zeros:0
X
W_c2/Adam_1:0W_c2/Adam_1/AssignW_c2/Adam_1/read:02W_c2/Adam_1/Initializer/zeros:0
P
b_c2/Adam:0b_c2/Adam/Assignb_c2/Adam/read:02b_c2/Adam/Initializer/zeros:0
X
b_c2/Adam_1:0b_c2/Adam_1/Assignb_c2/Adam_1/read:02b_c2/Adam_1/Initializer/zeros:0
P
W_c3/Adam:0W_c3/Adam/AssignW_c3/Adam/read:02W_c3/Adam/Initializer/zeros:0
X
W_c3/Adam_1:0W_c3/Adam_1/AssignW_c3/Adam_1/read:02W_c3/Adam_1/Initializer/zeros:0
P
b_c3/Adam:0b_c3/Adam/Assignb_c3/Adam/read:02b_c3/Adam/Initializer/zeros:0
X
b_c3/Adam_1:0b_c3/Adam_1/Assignb_c3/Adam_1/read:02b_c3/Adam_1/Initializer/zeros:0
P
W_c4/Adam:0W_c4/Adam/AssignW_c4/Adam/read:02W_c4/Adam/Initializer/zeros:0
X
W_c4/Adam_1:0W_c4/Adam_1/AssignW_c4/Adam_1/read:02W_c4/Adam_1/Initializer/zeros:0
P
b_c4/Adam:0b_c4/Adam/Assignb_c4/Adam/read:02b_c4/Adam/Initializer/zeros:0
X
b_c4/Adam_1:0b_c4/Adam_1/Assignb_c4/Adam_1/read:02b_c4/Adam_1/Initializer/zeros:0
P
W_c5/Adam:0W_c5/Adam/AssignW_c5/Adam/read:02W_c5/Adam/Initializer/zeros:0
X
W_c5/Adam_1:0W_c5/Adam_1/AssignW_c5/Adam_1/read:02W_c5/Adam_1/Initializer/zeros:0
P
b_c5/Adam:0b_c5/Adam/Assignb_c5/Adam/read:02b_c5/Adam/Initializer/zeros:0
X
b_c5/Adam_1:0b_c5/Adam_1/Assignb_c5/Adam_1/read:02b_c5/Adam_1/Initializer/zeros:0
P
W_n1/Adam:0W_n1/Adam/AssignW_n1/Adam/read:02W_n1/Adam/Initializer/zeros:0
X
W_n1/Adam_1:0W_n1/Adam_1/AssignW_n1/Adam_1/read:02W_n1/Adam_1/Initializer/zeros:0
P
b_n1/Adam:0b_n1/Adam/Assignb_n1/Adam/read:02b_n1/Adam/Initializer/zeros:0
X
b_n1/Adam_1:0b_n1/Adam_1/Assignb_n1/Adam_1/read:02b_n1/Adam_1/Initializer/zeros:0*?
serving_default?
 
X
X:0??????????
bc1
b_c1:0e
bc2
b_c2:0e
bc3
b_c3:0e
bc4
b_c4:0e
bc5
b_c5:0e
wn1
W_n1:0e
)
logits
logits:0?????????
#
wc1
W_c1:0e#
wc2
W_c2:0e#
wc3
W_c3:0e#
wc4
W_c4:0e#
wc5
W_c5:0e
bn1
b_n1:0e
tensorflow/serving/predict