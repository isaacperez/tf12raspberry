Āv
ķ
:
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T" 
Ttype:
2	"
use_lockingbool( 

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
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
B
Equal
x"T
y"T
z
"
Ttype:
2	

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

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
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
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

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
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.11.02v1.11.0-rc2-4-gc19e29306cĪ]
f
XPlaceholder*
dtype0*(
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
f
zeros/shape_as_tensorConst*
valueB"  
   *
dtype0*
_output_shapes
:
P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
zerosFillzeros/shape_as_tensorzeros/Const*
T0*

index_type0*
_output_shapes
:	

w
W
VariableV2*
shape:	
*
shared_name *
dtype0*
	container *
_output_shapes
:	


W/AssignAssignWzeros*
use_locking(*
T0*
_class

loc:@W*
validate_shape(*
_output_shapes
:	

U
W/readIdentityW*
T0*
_class

loc:@W*
_output_shapes
:	

T
zeros_1Const*
valueB
*    *
dtype0*
_output_shapes
:

m
B
VariableV2*
shape:
*
shared_name *
dtype0*
	container *
_output_shapes
:


B/AssignAssignBzeros_1*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class

loc:@B
P
B/readIdentityB*
_output_shapes
:
*
T0*
_class

loc:@B
s
MatMulMatMulXW/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:’’’’’’’’’

L
addAddMatMulB/read*'
_output_shapes
:’’’’’’’’’
*
T0
I
logitsIdentityadd*
T0*'
_output_shapes
:’’’’’’’’’

L
SoftmaxSoftmaxlogits*
T0*'
_output_shapes
:’’’’’’’’’

n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:’’’’’’’’’
*
shape:’’’’’’’’’

E
LogLogSoftmax*
T0*'
_output_shapes
:’’’’’’’’’

N
mulMulPlaceholderLog*
T0*'
_output_shapes
:’’’’’’’’’

V
ConstConst*
dtype0*
_output_shapes
:*
valueB"       
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
 *  ?*
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

gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
[
gradients/Sum_grad/ShapeShapemul*
_output_shapes
:*
T0*
out_type0

gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*
T0*'
_output_shapes
:’’’’’’’’’
*

Tmultiples0
c
gradients/mul_grad/ShapeShapePlaceholder*
_output_shapes
:*
T0*
out_type0
]
gradients/mul_grad/Shape_1ShapeLog*
T0*
out_type0*
_output_shapes
:
“
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
m
gradients/mul_grad/MulMulgradients/Sum_grad/TileLog*
T0*'
_output_shapes
:’’’’’’’’’


gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

w
gradients/mul_grad/Mul_1MulPlaceholdergradients/Sum_grad/Tile*'
_output_shapes
:’’’’’’’’’
*
T0
„
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*'
_output_shapes
:’’’’’’’’’
*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
Ś
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*'
_output_shapes
:’’’’’’’’’

ą
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’


gradients/Log_grad/Reciprocal
ReciprocalSoftmax.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:’’’’’’’’’


gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:’’’’’’’’’

t
gradients/Softmax_grad/mulMulgradients/Log_grad/mulSoftmax*
T0*'
_output_shapes
:’’’’’’’’’

w
,gradients/Softmax_grad/Sum/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
ŗ
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*
T0*'
_output_shapes
:’’’’’’’’’*

Tidx0*
	keep_dims(

gradients/Softmax_grad/subSubgradients/Log_grad/mulgradients/Softmax_grad/Sum*
T0*'
_output_shapes
:’’’’’’’’’

z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:’’’’’’’’’

^
gradients/add_grad/ShapeShapeMatMul*
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
“
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
„
gradients/add_grad/SumSumgradients/Softmax_grad/mul_1(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*'
_output_shapes
:’’’’’’’’’
*
T0*
Tshape0
©
gradients/add_grad/Sum_1Sumgradients/Softmax_grad/mul_1*gradients/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ś
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*'
_output_shapes
:’’’’’’’’’

Ó
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:

“
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyW/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:’’’’’’’’’
Ø
gradients/MatMul_grad/MatMul_1MatMulX+gradients/add_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	
*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
å
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*(
_output_shapes
:’’’’’’’’’*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
ā
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes
:	

b
GradientDescent/learning_rateConst*
valueB
 *
×#<*
dtype0*
_output_shapes
: 
ģ
-GradientDescent/update_W/ApplyGradientDescentApplyGradientDescentWGradientDescent/learning_rate0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class

loc:@W*
_output_shapes
:	

ä
-GradientDescent/update_B/ApplyGradientDescentApplyGradientDescentBGradientDescent/learning_rate-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class

loc:@B*
_output_shapes
:

w
GradientDescentNoOp.^GradientDescent/update_B/ApplyGradientDescent.^GradientDescent/update_W/ApplyGradientDescent
"
initNoOp	^B/Assign	^W/Assign
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
x
ArgMaxArgMaxSoftmaxArgMax/dimension*
output_type0	*#
_output_shapes
:’’’’’’’’’*

Tidx0*
T0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMax_1ArgMaxPlaceholderArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:’’’’’’’’’*

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:’’’’’’’’’
`
CastCastEqual*

SrcT0
*
Truncate( *

DstT0*#
_output_shapes
:’’’’’’’’’
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Y
MeanMeanCastConst_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_f8421850179c4d919677bcaee303337b/part*
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
e
save/SaveV2/tensor_namesConst*
valueBBBBW*
dtype0*
_output_shapes
:
g
save/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0*
_output_shapes
:
{
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesBW*
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

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
h
save/RestoreV2/tensor_namesConst*
valueBBBBW*
dtype0*
_output_shapes
:
j
save/RestoreV2/shape_and_slicesConst*
valueBB B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
_output_shapes

::*
dtypes
2

save/AssignAssignBsave/RestoreV2*
use_locking(*
T0*
_class

loc:@B*
validate_shape(*
_output_shapes
:


save/Assign_1AssignWsave/RestoreV2:1*
T0*
_class

loc:@W*
validate_shape(*
_output_shapes
:	
*
use_locking(
8
save/restore_shardNoOp^save/Assign^save/Assign_1
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"g
trainable_variablesPN
$
W:0W/AssignW/read:02zeros:08
&
B:0B/AssignB/read:02	zeros_1:08"
train_op

GradientDescent"]
	variablesPN
$
W:0W/AssignW/read:02zeros:08
&
B:0B/AssignB/read:02	zeros_1:08*Ŗ
serving_default
 
X
X:0’’’’’’’’’
B
B:0e
)
logits
logits:0’’’’’’’’’

W
W:0e	
tensorflow/serving/predict