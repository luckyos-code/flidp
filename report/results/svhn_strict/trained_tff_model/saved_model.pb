��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.14.12v2.14.0-10-g99d80a9e2548��
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
h

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
h

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
h

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_3
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0	
b
total_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0	
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:
*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	�@*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
�
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
�
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
�
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: *
dtype0
h

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_4
a
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
: *
dtype0
h

Variable_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_5
a
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
: *
dtype0
h

Variable_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_6
a
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
: *
dtype0
h

Variable_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_7
a
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
: *
dtype0
h

Variable_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_8
a
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
: *
dtype0
�
serving_default_args_0Placeholder*/
_output_shapes
:���������  *
dtype0	*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0conv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8� *-
f(R&
$__inference_signature_wrapper_507190

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
tff_trainable_variables
tff_non_trainable_variables
tff_local_variables
#forward_pass_training_type_spec
$ forward_pass_inference_type_spec
'#predict_on_batch_training_type_spec
($predict_on_batch_inference_type_spec
 serialized_metric_finalizers
	serialized_input_spec

flat_forward_pass_inference
flat_forward_pass_training
predict_on_batch_inference
predict_on_batch_training
$ report_local_unfinalized_metrics
reset_metrics

signatures*
J
0
1
2
3
4
5
6
7
8
9*
* 
.
0
1
2
3
4
 5*
^X
VARIABLE_VALUE
Variable_8:forward_pass_training_type_spec/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUE
Variable_7;forward_pass_inference_type_spec/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE
Variable_6>predict_on_batch_training_type_spec/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE
Variable_5?predict_on_batch_inference_type_spec/.ATTRIBUTES/VARIABLE_VALUE*
R
!sparse_categorical_accuracy
"loss
#num_examples
$num_batches*
TN
VARIABLE_VALUE
Variable_40serialized_input_spec/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 

%serving_default* 
]W
VARIABLE_VALUEconv2d_3/kernel4tff_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4tff_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_4/kernel4tff_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_4/bias4tff_trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_5/kernel4tff_trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_5/bias4tff_trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_2/kernel4tff_trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4tff_trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_3/kernel4tff_trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4tff_trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEtotal_30tff_local_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEcount_10tff_local_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEtotal_20tff_local_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEcount0tff_local_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEtotal_10tff_local_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEtotal0tff_local_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE
Variable_3Sserialized_metric_finalizers/sparse_categorical_accuracy/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE
Variable_2<serialized_metric_finalizers/loss/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE
Variable_1Dserialized_metric_finalizers/num_examples/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEVariableCserialized_metric_finalizers/num_batches/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4conv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biastotal_3count_1total_2counttotal_1total
Variable_3
Variable_2
Variable_1VariableConst*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *(
f#R!
__inference__traced_save_507362
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4conv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biastotal_3count_1total_2counttotal_1total
Variable_3
Variable_2
Variable_1Variable*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *+
f&R$
"__inference__traced_restore_507446��
�
�
3__inference_report_local_unfinalized_metrics_507027&
read_readvariableop_resource: (
read_1_readvariableop_resource: (
read_2_readvariableop_resource: (
read_3_readvariableop_resource: (
read_4_readvariableop_resource:	 (
read_5_readvariableop_resource:	 

identity_6

identity_7

identity_8	

identity_9	
identity_10
identity_11��Read/ReadVariableOp�Read_1/ReadVariableOp�Read_2/ReadVariableOp�Read_3/ReadVariableOp�Read_4/ReadVariableOp�Read_5/ReadVariableOph
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
: *
dtype0R
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
: l
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
: *
dtype0V

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: l
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
: *
dtype0V

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
: l
Read_3/ReadVariableOpReadVariableOpread_3_readvariableop_resource*
_output_shapes
: *
dtype0V

Identity_3IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
: l
Read_4/ReadVariableOpReadVariableOpread_4_readvariableop_resource*
_output_shapes
: *
dtype0	V

Identity_4IdentityRead_4/ReadVariableOp:value:0*
T0	*
_output_shapes
: l
Read_5/ReadVariableOpReadVariableOpread_5_readvariableop_resource*
_output_shapes
: *
dtype0	V

Identity_5IdentityRead_5/ReadVariableOp:value:0*
T0	*
_output_shapes
: S

Identity_6IdentityIdentity_2:output:0^NoOp*
T0*
_output_shapes
: S

Identity_7IdentityIdentity_3:output:0^NoOp*
T0*
_output_shapes
: S

Identity_8IdentityIdentity_5:output:0^NoOp*
T0	*
_output_shapes
: S

Identity_9IdentityIdentity_4:output:0^NoOp*
T0	*
_output_shapes
: R
Identity_10IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: T
Identity_11IdentityIdentity_1:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp^Read_3/ReadVariableOp^Read_4/ReadVariableOp^Read_5/ReadVariableOp*
_output_shapes
 "#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp2.
Read_3/ReadVariableOpRead_3/ReadVariableOp2.
Read_4/ReadVariableOpRead_4/ReadVariableOp2.
Read_5/ReadVariableOpRead_5/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource
�
�
#__inference_flattened_output_506945

args_0	!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5:	�@
	unknown_6:@
	unknown_7:@

	unknown_8:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8� *,
f'R%
#__inference_predict_on_batch_506416o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&
"
 
_user_specified_name506941:&	"
 
_user_specified_name506939:&"
 
_user_specified_name506937:&"
 
_user_specified_name506935:&"
 
_user_specified_name506933:&"
 
_user_specified_name506931:&"
 
_user_specified_name506929:&"
 
_user_specified_name506927:&"
 
_user_specified_name506925:&"
 
_user_specified_name506923:W S
/
_output_shapes
:���������  
 
_user_specified_nameargs_0
�r
�
"__inference__traced_restore_507446
file_prefix%
assignvariableop_variable_8: '
assignvariableop_1_variable_7: '
assignvariableop_2_variable_6: '
assignvariableop_3_variable_5: '
assignvariableop_4_variable_4: <
"assignvariableop_5_conv2d_3_kernel: .
 assignvariableop_6_conv2d_3_bias: <
"assignvariableop_7_conv2d_4_kernel:  .
 assignvariableop_8_conv2d_4_bias: <
"assignvariableop_9_conv2d_5_kernel:  /
!assignvariableop_10_conv2d_5_bias: 5
"assignvariableop_11_dense_2_kernel:	�@.
 assignvariableop_12_dense_2_bias:@4
"assignvariableop_13_dense_3_kernel:@
.
 assignvariableop_14_dense_3_bias:
%
assignvariableop_15_total_3: %
assignvariableop_16_count_1: %
assignvariableop_17_total_2: #
assignvariableop_18_count: %
assignvariableop_19_total_1:	 #
assignvariableop_20_total:	 (
assignvariableop_21_variable_3: (
assignvariableop_22_variable_2: (
assignvariableop_23_variable_1: &
assignvariableop_24_variable: 
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:forward_pass_training_type_spec/.ATTRIBUTES/VARIABLE_VALUEB;forward_pass_inference_type_spec/.ATTRIBUTES/VARIABLE_VALUEB>predict_on_batch_training_type_spec/.ATTRIBUTES/VARIABLE_VALUEB?predict_on_batch_inference_type_spec/.ATTRIBUTES/VARIABLE_VALUEB0serialized_input_spec/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/5/.ATTRIBUTES/VARIABLE_VALUEBSserialized_metric_finalizers/sparse_categorical_accuracy/.ATTRIBUTES/VARIABLE_VALUEB<serialized_metric_finalizers/loss/.ATTRIBUTES/VARIABLE_VALUEBDserialized_metric_finalizers/num_examples/.ATTRIBUTES/VARIABLE_VALUEBCserialized_metric_finalizers/num_batches/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_8Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_7Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_6Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_5Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_4Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_3_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_3_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_4_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_4_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_5_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_5_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_2_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp assignvariableop_12_dense_2_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_3_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp assignvariableop_14_dense_3_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_3Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_2Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_3Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_2Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_1Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variableIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_26Identity_26:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:%!

_user_specified_nametotal:'#
!
_user_specified_name	total_1:%!

_user_specified_namecount:'#
!
_user_specified_name	total_2:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_3:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:-)
'
_user_specified_nameconv2d_5/bias:/
+
)
_user_specified_nameconv2d_5/kernel:-	)
'
_user_specified_nameconv2d_4/bias:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv2d_3/bias:/+
)
_user_specified_nameconv2d_3/kernel:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�s
�	
#__inference_predict_on_batch_506416
x	N
4sequential_1_conv2d_3_conv2d_readvariableop_resource: C
5sequential_1_conv2d_3_biasadd_readvariableop_resource: N
4sequential_1_conv2d_4_conv2d_readvariableop_resource:  C
5sequential_1_conv2d_4_biasadd_readvariableop_resource: N
4sequential_1_conv2d_5_conv2d_readvariableop_resource:  C
5sequential_1_conv2d_5_biasadd_readvariableop_resource: F
3sequential_1_dense_2_matmul_readvariableop_resource:	�@B
4sequential_1_dense_2_biasadd_readvariableop_resource:@E
3sequential_1_dense_3_matmul_readvariableop_resource:@
B
4sequential_1_dense_3_biasadd_readvariableop_resource:

identity��,sequential_1/conv2d_3/BiasAdd/ReadVariableOp�+sequential_1/conv2d_3/Conv2D/ReadVariableOp�,sequential_1/conv2d_4/BiasAdd/ReadVariableOp�+sequential_1/conv2d_4/Conv2D/ReadVariableOp�,sequential_1/conv2d_5/BiasAdd/ReadVariableOp�+sequential_1/conv2d_5/Conv2D/ReadVariableOp�+sequential_1/dense_2/BiasAdd/ReadVariableOp�*sequential_1/dense_2/MatMul/ReadVariableOp�+sequential_1/dense_3/BiasAdd/ReadVariableOp�*sequential_1/dense_3/MatMul/ReadVariableOpe
sequential_1/CastCastx*

DstT0*

SrcT0	*/
_output_shapes
:���������  d
sequential_1/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;f
!sequential_1/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_1/rescaling_1/mulMulsequential_1/Cast:y:0(sequential_1/rescaling_1/Cast/x:output:0*
T0*/
_output_shapes
:���������  �
sequential_1/rescaling_1/addAddV2 sequential_1/rescaling_1/mul:z:0*sequential_1/rescaling_1/Cast_1/x:output:0*
T0*/
_output_shapes
:���������  �
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
sequential_1/conv2d_3/Conv2DConv2D sequential_1/rescaling_1/add:z:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
sequential_1/conv2d_3/ReluRelu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
$sequential_1/max_pooling2d_3/MaxPoolMaxPool(sequential_1/conv2d_3/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
i
$sequential_1/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
"sequential_1/dropout_4/dropout/MulMul-sequential_1/max_pooling2d_3/MaxPool:output:0-sequential_1/dropout_4/dropout/Const:output:0*
T0*/
_output_shapes
:��������� �
$sequential_1/dropout_4/dropout/ShapeShape-sequential_1/max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
::���
;sequential_1/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_4/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0r
-sequential_1/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
+sequential_1/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� k
&sequential_1/dropout_4/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'sequential_1/dropout_4/dropout/SelectV2SelectV2/sequential_1/dropout_4/dropout/GreaterEqual:z:0&sequential_1/dropout_4/dropout/Mul:z:0/sequential_1/dropout_4/dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� �
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
sequential_1/conv2d_4/Conv2DConv2D0sequential_1/dropout_4/dropout/SelectV2:output:03sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
sequential_1/conv2d_4/ReluRelu&sequential_1/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
$sequential_1/max_pooling2d_4/MaxPoolMaxPool(sequential_1/conv2d_4/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
i
$sequential_1/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
"sequential_1/dropout_5/dropout/MulMul-sequential_1/max_pooling2d_4/MaxPool:output:0-sequential_1/dropout_5/dropout/Const:output:0*
T0*/
_output_shapes
:��������� �
$sequential_1/dropout_5/dropout/ShapeShape-sequential_1/max_pooling2d_4/MaxPool:output:0*
T0*
_output_shapes
::���
;sequential_1/dropout_5/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_5/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0r
-sequential_1/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
+sequential_1/dropout_5/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_5/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_5/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� k
&sequential_1/dropout_5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'sequential_1/dropout_5/dropout/SelectV2SelectV2/sequential_1/dropout_5/dropout/GreaterEqual:z:0&sequential_1/dropout_5/dropout/Mul:z:0/sequential_1/dropout_5/dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� �
+sequential_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
sequential_1/conv2d_5/Conv2DConv2D0sequential_1/dropout_5/dropout/SelectV2:output:03sequential_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
,sequential_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_1/conv2d_5/BiasAddBiasAdd%sequential_1/conv2d_5/Conv2D:output:04sequential_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
sequential_1/conv2d_5/ReluRelu&sequential_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
$sequential_1/max_pooling2d_5/MaxPoolMaxPool(sequential_1/conv2d_5/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
i
$sequential_1/dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
"sequential_1/dropout_6/dropout/MulMul-sequential_1/max_pooling2d_5/MaxPool:output:0-sequential_1/dropout_6/dropout/Const:output:0*
T0*/
_output_shapes
:��������� �
$sequential_1/dropout_6/dropout/ShapeShape-sequential_1/max_pooling2d_5/MaxPool:output:0*
T0*
_output_shapes
::���
;sequential_1/dropout_6/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_6/dropout/Shape:output:0*
T0*/
_output_shapes
:��������� *
dtype0r
-sequential_1/dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
+sequential_1/dropout_6/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_6/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_6/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:��������� k
&sequential_1/dropout_6/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'sequential_1/dropout_6/dropout/SelectV2SelectV2/sequential_1/dropout_6/dropout/GreaterEqual:z:0&sequential_1/dropout_6/dropout/Mul:z:0/sequential_1/dropout_6/dropout/Const_1:output:0*
T0*/
_output_shapes
:��������� m
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
sequential_1/flatten_1/ReshapeReshape0sequential_1/dropout_6/dropout/SelectV2:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:�����������
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential_1/dense_2/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@i
$sequential_1/dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
"sequential_1/dropout_7/dropout/MulMul'sequential_1/dense_2/Relu:activations:0-sequential_1/dropout_7/dropout/Const:output:0*
T0*'
_output_shapes
:���������@�
$sequential_1/dropout_7/dropout/ShapeShape'sequential_1/dense_2/Relu:activations:0*
T0*
_output_shapes
::���
;sequential_1/dropout_7/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_7/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0r
-sequential_1/dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
+sequential_1/dropout_7/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_7/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_7/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@k
&sequential_1/dropout_7/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
'sequential_1/dropout_7/dropout/SelectV2SelectV2/sequential_1/dropout_7/dropout/GreaterEqual:z:0&sequential_1/dropout_7/dropout/Mul:z:0/sequential_1/dropout_7/dropout/Const_1:output:0*
T0*'
_output_shapes
:���������@�
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0�
sequential_1/dense_3/MatMulMatMul0sequential_1/dropout_7/dropout/SelectV2:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������
u
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp-^sequential_1/conv2d_5/BiasAdd/ReadVariableOp,^sequential_1/conv2d_5/Conv2D/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  : : : : : : : : : : 2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_5/BiasAdd/ReadVariableOp,sequential_1/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_5/Conv2D/ReadVariableOp+sequential_1/conv2d_5/Conv2D/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:R N
/
_output_shapes
:���������  

_user_specified_namex
�
�
#__inference_flattened_output_506998

args_0	!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5:	�@
	unknown_6:@
	unknown_7:@

	unknown_8:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8� *,
f'R%
#__inference_predict_on_batch_506685o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&
"
 
_user_specified_name506994:&	"
 
_user_specified_name506992:&"
 
_user_specified_name506990:&"
 
_user_specified_name506988:&"
 
_user_specified_name506986:&"
 
_user_specified_name506984:&"
 
_user_specified_name506982:&"
 
_user_specified_name506980:&"
 
_user_specified_name506978:&"
 
_user_specified_name506976:W S
/
_output_shapes
:���������  
 
_user_specified_nameargs_0
�
�
 __inference_reset_metrics_507164#
assignvariableop_resource: %
assignvariableop_1_resource: %
assignvariableop_2_resource: %
assignvariableop_3_resource: %
assignvariableop_4_resource:	 %
assignvariableop_5_resource:	 ��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOpAssignVariableOpassignvariableop_resourceConst:output:0*
_output_shapes
 *
dtype0*
validate_shape(L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOp_1AssignVariableOpassignvariableop_1_resourceConst_1:output:0*
_output_shapes
 *
dtype0*
validate_shape(L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOp_2AssignVariableOpassignvariableop_2_resourceConst_2:output:0*
_output_shapes
 *
dtype0*
validate_shape(L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *    �
AssignVariableOp_3AssignVariableOpassignvariableop_3_resourceConst_3:output:0*
_output_shapes
 *
dtype0*
validate_shape(I
Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R �
AssignVariableOp_4AssignVariableOpassignvariableop_4_resourceConst_4:output:0*
_output_shapes
 *
dtype0	*
validate_shape(I
Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R �
AssignVariableOp_5AssignVariableOpassignvariableop_5_resourceConst_5:output:0*
_output_shapes
 *
dtype0	*
validate_shape(*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:( $
"
_user_specified_name
resource
��
�
__inference__traced_save_507362
file_prefix+
!read_disablecopyonread_variable_8: -
#read_1_disablecopyonread_variable_7: -
#read_2_disablecopyonread_variable_6: -
#read_3_disablecopyonread_variable_5: -
#read_4_disablecopyonread_variable_4: B
(read_5_disablecopyonread_conv2d_3_kernel: 4
&read_6_disablecopyonread_conv2d_3_bias: B
(read_7_disablecopyonread_conv2d_4_kernel:  4
&read_8_disablecopyonread_conv2d_4_bias: B
(read_9_disablecopyonread_conv2d_5_kernel:  5
'read_10_disablecopyonread_conv2d_5_bias: ;
(read_11_disablecopyonread_dense_2_kernel:	�@4
&read_12_disablecopyonread_dense_2_bias:@:
(read_13_disablecopyonread_dense_3_kernel:@
4
&read_14_disablecopyonread_dense_3_bias:
+
!read_15_disablecopyonread_total_3: +
!read_16_disablecopyonread_count_1: +
!read_17_disablecopyonread_total_2: )
read_18_disablecopyonread_count: +
!read_19_disablecopyonread_total_1:	 )
read_20_disablecopyonread_total:	 .
$read_21_disablecopyonread_variable_3: .
$read_22_disablecopyonread_variable_2: .
$read_23_disablecopyonread_variable_1: ,
"read_24_disablecopyonread_variable: 
savev2_const
identity_51��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: s
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_8"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_8^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
: w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_variable_7"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_variable_7^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: w
Read_2/DisableCopyOnReadDisableCopyOnRead#read_2_disablecopyonread_variable_6"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp#read_2_disablecopyonread_variable_6^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: w
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_variable_5"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_variable_5^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: w
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_variable_4"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_variable_4^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_5/DisableCopyOnReadDisableCopyOnRead(read_5_disablecopyonread_conv2d_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp(read_5_disablecopyonread_conv2d_3_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0v
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*&
_output_shapes
: z
Read_6/DisableCopyOnReadDisableCopyOnRead&read_6_disablecopyonread_conv2d_3_bias"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp&read_6_disablecopyonread_conv2d_3_bias^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_conv2d_4_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0v
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*&
_output_shapes
:  z
Read_8/DisableCopyOnReadDisableCopyOnRead&read_8_disablecopyonread_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp&read_8_disablecopyonread_conv2d_4_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_9/DisableCopyOnReadDisableCopyOnRead(read_9_disablecopyonread_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp(read_9_disablecopyonread_conv2d_5_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0v
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*&
_output_shapes
:  |
Read_10/DisableCopyOnReadDisableCopyOnRead'read_10_disablecopyonread_conv2d_5_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp'read_10_disablecopyonread_conv2d_5_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: }
Read_11/DisableCopyOnReadDisableCopyOnRead(read_11_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp(read_11_disablecopyonread_dense_2_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@{
Read_12/DisableCopyOnReadDisableCopyOnRead&read_12_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp&read_12_disablecopyonread_dense_2_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_dense_3_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@
*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@
e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:@
{
Read_14/DisableCopyOnReadDisableCopyOnRead&read_14_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp&read_14_disablecopyonread_dense_3_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:
v
Read_15/DisableCopyOnReadDisableCopyOnRead!read_15_disablecopyonread_total_3"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp!read_15_disablecopyonread_total_3^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_16/DisableCopyOnReadDisableCopyOnRead!read_16_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp!read_16_disablecopyonread_count_1^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_17/DisableCopyOnReadDisableCopyOnRead!read_17_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp!read_17_disablecopyonread_total_2^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_18/DisableCopyOnReadDisableCopyOnReadread_18_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpread_18_disablecopyonread_count^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_19/DisableCopyOnReadDisableCopyOnRead!read_19_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp!read_19_disablecopyonread_total_1^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0	*
_output_shapes
: t
Read_20/DisableCopyOnReadDisableCopyOnReadread_20_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOpread_20_disablecopyonread_total^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0	*
_output_shapes
: y
Read_21/DisableCopyOnReadDisableCopyOnRead$read_21_disablecopyonread_variable_3"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp$read_21_disablecopyonread_variable_3^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_22/DisableCopyOnReadDisableCopyOnRead$read_22_disablecopyonread_variable_2"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp$read_22_disablecopyonread_variable_2^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: y
Read_23/DisableCopyOnReadDisableCopyOnRead$read_23_disablecopyonread_variable_1"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp$read_23_disablecopyonread_variable_1^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: w
Read_24/DisableCopyOnReadDisableCopyOnRead"read_24_disablecopyonread_variable"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp"read_24_disablecopyonread_variable^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B:forward_pass_training_type_spec/.ATTRIBUTES/VARIABLE_VALUEB;forward_pass_inference_type_spec/.ATTRIBUTES/VARIABLE_VALUEB>predict_on_batch_training_type_spec/.ATTRIBUTES/VARIABLE_VALUEB?predict_on_batch_inference_type_spec/.ATTRIBUTES/VARIABLE_VALUEB0serialized_input_spec/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB4tff_trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0tff_local_variables/5/.ATTRIBUTES/VARIABLE_VALUEBSserialized_metric_finalizers/sparse_categorical_accuracy/.ATTRIBUTES/VARIABLE_VALUEB<serialized_metric_finalizers/loss/.ATTRIBUTES/VARIABLE_VALUEBDserialized_metric_finalizers/num_examples/.ATTRIBUTES/VARIABLE_VALUEBCserialized_metric_finalizers/num_batches/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *(
dtypes
2		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_50Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_51IdentityIdentity_50:output:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_51Identity_51:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:%!

_user_specified_nametotal:'#
!
_user_specified_name	total_1:%!

_user_specified_namecount:'#
!
_user_specified_name	total_2:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_3:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:-)
'
_user_specified_nameconv2d_5/bias:/
+
)
_user_specified_nameconv2d_5/kernel:-	)
'
_user_specified_nameconv2d_4/bias:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv2d_3/bias:/+
)
_user_specified_nameconv2d_3/kernel:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
$__inference_signature_wrapper_507190

args_0	!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5:	�@
	unknown_6:@
	unknown_7:@

	unknown_8:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8� *,
f'R%
#__inference_flattened_output_506998o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&
"
 
_user_specified_name507186:&	"
 
_user_specified_name507184:&"
 
_user_specified_name507182:&"
 
_user_specified_name507180:&"
 
_user_specified_name507178:&"
 
_user_specified_name507176:&"
 
_user_specified_name507174:&"
 
_user_specified_name507172:&"
 
_user_specified_name507170:&"
 
_user_specified_name507168:W S
/
_output_shapes
:���������  
 
_user_specified_nameargs_0
ȁ
�
__inference_forward_pass_506542
batch_input	
batch_input_1	!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5:	�@
	unknown_6:@
	unknown_7:@

	unknown_8:
&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: (
assignaddvariableop_4_resource:	 (
assignaddvariableop_5_resource:	 
identity

identity_1

identity_2��AssignAddVariableOp�AssignAddVariableOp_1�AssignAddVariableOp_2�AssignAddVariableOp_3�AssignAddVariableOp_4�AssignAddVariableOp_5�StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallbatch_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8� *,
f'R%
#__inference_predict_on_batch_506416j
%sparse_categorical_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3j
%sparse_categorical_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#sparse_categorical_crossentropy/subSub.sparse_categorical_crossentropy/sub/x:output:0.sparse_categorical_crossentropy/Const:output:0*
T0*
_output_shapes
: �
5sparse_categorical_crossentropy/clip_by_value/MinimumMinimum StatefulPartitionedCall:output:0'sparse_categorical_crossentropy/sub:z:0*
T0*'
_output_shapes
:���������
�
-sparse_categorical_crossentropy/clip_by_valueMaximum9sparse_categorical_crossentropy/clip_by_value/Minimum:z:0.sparse_categorical_crossentropy/Const:output:0*
T0*'
_output_shapes
:���������
�
#sparse_categorical_crossentropy/LogLog1sparse_categorical_crossentropy/clip_by_value:z:0*
T0*'
_output_shapes
:���������
�
%sparse_categorical_crossentropy/ShapeShape'sparse_categorical_crossentropy/Log:y:0*
T0*
_output_shapes
::���
Isparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ShapeShapebatch_input_1*
T0	*
_output_shapes
::���
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits'sparse_categorical_crossentropy/Log:y:0batch_input_1*
T0*6
_output_shapes$
":���������:���������
x
3sparse_categorical_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
1sparse_categorical_crossentropy/weighted_loss/MulMulnsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:loss:0<sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*#
_output_shapes
:���������
5sparse_categorical_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
1sparse_categorical_crossentropy/weighted_loss/SumSum5sparse_categorical_crossentropy/weighted_loss/Mul:z:0>sparse_categorical_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: �
:sparse_categorical_crossentropy/weighted_loss/num_elementsSize5sparse_categorical_crossentropy/weighted_loss/Mul:z:0*
T0*
_output_shapes
: �
?sparse_categorical_crossentropy/weighted_loss/num_elements/CastCastCsparse_categorical_crossentropy/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: t
2sparse_categorical_crossentropy/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : {
9sparse_categorical_crossentropy/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : {
9sparse_categorical_crossentropy/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
3sparse_categorical_crossentropy/weighted_loss/rangeRangeBsparse_categorical_crossentropy/weighted_loss/range/start:output:0;sparse_categorical_crossentropy/weighted_loss/Rank:output:0Bsparse_categorical_crossentropy/weighted_loss/range/delta:output:0*
_output_shapes
: �
3sparse_categorical_crossentropy/weighted_loss/Sum_1Sum:sparse_categorical_crossentropy/weighted_loss/Sum:output:0<sparse_categorical_crossentropy/weighted_loss/range:output:0*
T0*
_output_shapes
: �
3sparse_categorical_crossentropy/weighted_loss/valueDivNoNan<sparse_categorical_crossentropy/weighted_loss/Sum_1:output:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: X
CastCastbatch_input_1*

DstT0*

SrcT0	*#
_output_shapes
:���������K
ShapeShapeCast:y:0*
T0*
_output_shapes
::��[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������{
ArgMaxArgMax StatefulPartitionedCall:output:0ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������\
Cast_1CastArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������R
EqualEqualCast:y:0
Cast_1:y:0*
T0*#
_output_shapes
:���������V
Cast_2Cast	Equal:z:0*

DstT0*

SrcT0
*#
_output_shapes
:���������O
ConstConst*
_output_shapes
:*
dtype0*
valueB: o
SumSum
Cast_2:y:0Const:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: �
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype09
SizeSize
Cast_2:y:0*
T0*
_output_shapes
: M
Cast_3CastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resource
Cast_3:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0e
Shape_1Shape StatefulPartitionedCall:output:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
'sparse_categorical_crossentropy_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3l
'sparse_categorical_crossentropy_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%sparse_categorical_crossentropy_1/subSub0sparse_categorical_crossentropy_1/sub/x:output:00sparse_categorical_crossentropy_1/Const:output:0*
T0*
_output_shapes
: �
7sparse_categorical_crossentropy_1/clip_by_value/MinimumMinimum StatefulPartitionedCall:output:0)sparse_categorical_crossentropy_1/sub:z:0*
T0*'
_output_shapes
:���������
�
/sparse_categorical_crossentropy_1/clip_by_valueMaximum;sparse_categorical_crossentropy_1/clip_by_value/Minimum:z:00sparse_categorical_crossentropy_1/Const:output:0*
T0*'
_output_shapes
:���������
�
%sparse_categorical_crossentropy_1/LogLog3sparse_categorical_crossentropy_1/clip_by_value:z:0*
T0*'
_output_shapes
:���������
�
'sparse_categorical_crossentropy_1/ShapeShape)sparse_categorical_crossentropy_1/Log:y:0*
T0*
_output_shapes
::���
Ksparse_categorical_crossentropy_1/SparseSoftmaxCrossEntropyWithLogits/ShapeShapebatch_input_1*
T0	*
_output_shapes
::���
isparse_categorical_crossentropy_1/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits)sparse_categorical_crossentropy_1/Log:y:0batch_input_1*
T0*6
_output_shapes$
":���������:���������
z
5sparse_categorical_crossentropy_1/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3sparse_categorical_crossentropy_1/weighted_loss/MulMulpsparse_categorical_crossentropy_1/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:loss:0>sparse_categorical_crossentropy_1/weighted_loss/Const:output:0*
T0*#
_output_shapes
:����������
7sparse_categorical_crossentropy_1/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
3sparse_categorical_crossentropy_1/weighted_loss/SumSum7sparse_categorical_crossentropy_1/weighted_loss/Mul:z:0@sparse_categorical_crossentropy_1/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: �
<sparse_categorical_crossentropy_1/weighted_loss/num_elementsSize7sparse_categorical_crossentropy_1/weighted_loss/Mul:z:0*
T0*
_output_shapes
: �
Asparse_categorical_crossentropy_1/weighted_loss/num_elements/CastCastEsparse_categorical_crossentropy_1/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: v
4sparse_categorical_crossentropy_1/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : }
;sparse_categorical_crossentropy_1/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : }
;sparse_categorical_crossentropy_1/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
5sparse_categorical_crossentropy_1/weighted_loss/rangeRangeDsparse_categorical_crossentropy_1/weighted_loss/range/start:output:0=sparse_categorical_crossentropy_1/weighted_loss/Rank:output:0Dsparse_categorical_crossentropy_1/weighted_loss/range/delta:output:0*
_output_shapes
: �
5sparse_categorical_crossentropy_1/weighted_loss/Sum_1Sum<sparse_categorical_crossentropy_1/weighted_loss/Sum:output:0>sparse_categorical_crossentropy_1/weighted_loss/range:output:0*
T0*
_output_shapes
: �
5sparse_categorical_crossentropy_1/weighted_loss/valueDivNoNan>sparse_categorical_crossentropy_1/weighted_loss/Sum_1:output:0Esparse_categorical_crossentropy_1/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: V
Cast_4Caststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: r
MulMul9sparse_categorical_crossentropy_1/weighted_loss/value:z:0
Cast_4:y:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: n
Sum_1SumMul:z:0range:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: �
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceSum_1:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B : O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*
_output_shapes
: K
Sum_2Sum
Cast_4:y:0range_1:output:0*
T0*
_output_shapes
: �
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resourceSum_2:output:0^AssignAddVariableOp_2*
_output_shapes
 *
dtype0R
Shape_2Shapebatch_input_1*
T0	*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_5Caststrided_slice_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B : O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_2Rangerange_2/start:output:0Rank_2:output:0range_2/delta:output:0*
_output_shapes
: s
Sum_3Sum
Cast_5:y:0range_2:output:0*
T0	*&
 _has_manual_control_dependencies(*
_output_shapes
: 
AssignAddVariableOp_4AssignAddVariableOpassignaddvariableop_4_resourceSum_3:output:0*
_output_shapes
 *
dtype0	J
Cast_6/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_6CastCast_6/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B : O
range_3/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_3Rangerange_3/start:output:0Rank_3:output:0range_3/delta:output:0*
_output_shapes
: s
Sum_4Sum
Cast_6:y:0range_3:output:0*
T0	*&
 _has_manual_control_dependencies(*
_output_shapes
: 
AssignAddVariableOp_5AssignAddVariableOpassignaddvariableop_5_resourceSum_4:output:0*
_output_shapes
 *
dtype0	P
Shape_3Shapebatch_input*
T0	*
_output_shapes
::��_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_3:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
IdentityIdentity7sparse_categorical_crossentropy/weighted_loss/value:z:0^NoOp*
T0*
_output_shapes
: q

Identity_1Identity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
X

Identity_2Identitystrided_slice_2:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^AssignAddVariableOp_4^AssignAddVariableOp_5^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:���������  :���������: : : : : : : : : : : : : : : : 2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_32.
AssignAddVariableOp_4AssignAddVariableOp_42.
AssignAddVariableOp_5AssignAddVariableOp_52*
AssignAddVariableOpAssignAddVariableOp22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:&"
 
_user_specified_name506435:&
"
 
_user_specified_name506433:&	"
 
_user_specified_name506431:&"
 
_user_specified_name506429:&"
 
_user_specified_name506427:&"
 
_user_specified_name506425:&"
 
_user_specified_name506423:&"
 
_user_specified_name506421:&"
 
_user_specified_name506419:&"
 
_user_specified_name506417:PL
#
_output_shapes
:���������
%
_user_specified_namebatch_input:\ X
/
_output_shapes
:���������  
%
_user_specified_namebatch_input
�
�
#__inference_flattened_output_506623

args_0	
args_0_1	!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5:	�@
	unknown_6:@
	unknown_7:@

	unknown_8:

	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13:	 

unknown_14:	 
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
: :���������
: *,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8� *(
f#R!
__inference_forward_pass_506542^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
: <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:���������  :���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name506615:&"
 
_user_specified_name506613:&"
 
_user_specified_name506611:&"
 
_user_specified_name506609:&"
 
_user_specified_name506607:&"
 
_user_specified_name506605:&"
 
_user_specified_name506603:&
"
 
_user_specified_name506601:&	"
 
_user_specified_name506599:&"
 
_user_specified_name506597:&"
 
_user_specified_name506595:&"
 
_user_specified_name506593:&"
 
_user_specified_name506591:&"
 
_user_specified_name506589:&"
 
_user_specified_name506587:&"
 
_user_specified_name506585:KG
#
_output_shapes
:���������
 
_user_specified_nameargs_0:W S
/
_output_shapes
:���������  
 
_user_specified_nameargs_0
�M
�	
#__inference_predict_on_batch_506685
x	N
4sequential_1_conv2d_3_conv2d_readvariableop_resource: C
5sequential_1_conv2d_3_biasadd_readvariableop_resource: N
4sequential_1_conv2d_4_conv2d_readvariableop_resource:  C
5sequential_1_conv2d_4_biasadd_readvariableop_resource: N
4sequential_1_conv2d_5_conv2d_readvariableop_resource:  C
5sequential_1_conv2d_5_biasadd_readvariableop_resource: F
3sequential_1_dense_2_matmul_readvariableop_resource:	�@B
4sequential_1_dense_2_biasadd_readvariableop_resource:@E
3sequential_1_dense_3_matmul_readvariableop_resource:@
B
4sequential_1_dense_3_biasadd_readvariableop_resource:

identity��,sequential_1/conv2d_3/BiasAdd/ReadVariableOp�+sequential_1/conv2d_3/Conv2D/ReadVariableOp�,sequential_1/conv2d_4/BiasAdd/ReadVariableOp�+sequential_1/conv2d_4/Conv2D/ReadVariableOp�,sequential_1/conv2d_5/BiasAdd/ReadVariableOp�+sequential_1/conv2d_5/Conv2D/ReadVariableOp�+sequential_1/dense_2/BiasAdd/ReadVariableOp�*sequential_1/dense_2/MatMul/ReadVariableOp�+sequential_1/dense_3/BiasAdd/ReadVariableOp�*sequential_1/dense_3/MatMul/ReadVariableOpe
sequential_1/CastCastx*

DstT0*

SrcT0	*/
_output_shapes
:���������  d
sequential_1/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;f
!sequential_1/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_1/rescaling_1/mulMulsequential_1/Cast:y:0(sequential_1/rescaling_1/Cast/x:output:0*
T0*/
_output_shapes
:���������  �
sequential_1/rescaling_1/addAddV2 sequential_1/rescaling_1/mul:z:0*sequential_1/rescaling_1/Cast_1/x:output:0*
T0*/
_output_shapes
:���������  �
+sequential_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
sequential_1/conv2d_3/Conv2DConv2D sequential_1/rescaling_1/add:z:03sequential_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
,sequential_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_1/conv2d_3/BiasAddBiasAdd%sequential_1/conv2d_3/Conv2D:output:04sequential_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
sequential_1/conv2d_3/ReluRelu&sequential_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
$sequential_1/max_pooling2d_3/MaxPoolMaxPool(sequential_1/conv2d_3/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
�
sequential_1/dropout_4/IdentityIdentity-sequential_1/max_pooling2d_3/MaxPool:output:0*
T0*/
_output_shapes
:��������� �
+sequential_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
sequential_1/conv2d_4/Conv2DConv2D(sequential_1/dropout_4/Identity:output:03sequential_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
,sequential_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_1/conv2d_4/BiasAddBiasAdd%sequential_1/conv2d_4/Conv2D:output:04sequential_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
sequential_1/conv2d_4/ReluRelu&sequential_1/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
$sequential_1/max_pooling2d_4/MaxPoolMaxPool(sequential_1/conv2d_4/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
�
sequential_1/dropout_5/IdentityIdentity-sequential_1/max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:��������� �
+sequential_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
sequential_1/conv2d_5/Conv2DConv2D(sequential_1/dropout_5/Identity:output:03sequential_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
�
,sequential_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
sequential_1/conv2d_5/BiasAddBiasAdd%sequential_1/conv2d_5/Conv2D:output:04sequential_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
sequential_1/conv2d_5/ReluRelu&sequential_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
$sequential_1/max_pooling2d_5/MaxPoolMaxPool(sequential_1/conv2d_5/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
�
sequential_1/dropout_6/IdentityIdentity-sequential_1/max_pooling2d_5/MaxPool:output:0*
T0*/
_output_shapes
:��������� m
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
sequential_1/flatten_1/ReshapeReshape(sequential_1/dropout_6/Identity:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:�����������
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
sequential_1/dense_2/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
sequential_1/dropout_7/IdentityIdentity'sequential_1/dense_2/Relu:activations:0*
T0*'
_output_shapes
:���������@�
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype0�
sequential_1/dense_3/MatMulMatMul(sequential_1/dropout_7/Identity:output:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
sequential_1/dense_3/SoftmaxSoftmax%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������
u
IdentityIdentity&sequential_1/dense_3/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp-^sequential_1/conv2d_3/BiasAdd/ReadVariableOp,^sequential_1/conv2d_3/Conv2D/ReadVariableOp-^sequential_1/conv2d_4/BiasAdd/ReadVariableOp,^sequential_1/conv2d_4/Conv2D/ReadVariableOp-^sequential_1/conv2d_5/BiasAdd/ReadVariableOp,^sequential_1/conv2d_5/Conv2D/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������  : : : : : : : : : : 2\
,sequential_1/conv2d_3/BiasAdd/ReadVariableOp,sequential_1/conv2d_3/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_3/Conv2D/ReadVariableOp+sequential_1/conv2d_3/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_4/BiasAdd/ReadVariableOp,sequential_1/conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_4/Conv2D/ReadVariableOp+sequential_1/conv2d_4/Conv2D/ReadVariableOp2\
,sequential_1/conv2d_5/BiasAdd/ReadVariableOp,sequential_1/conv2d_5/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_5/Conv2D/ReadVariableOp+sequential_1/conv2d_5/Conv2D/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:R N
/
_output_shapes
:���������  

_user_specified_namex
ȁ
�
__inference_forward_pass_506811
batch_input	
batch_input_1	!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5:	�@
	unknown_6:@
	unknown_7:@

	unknown_8:
&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: (
assignaddvariableop_2_resource: (
assignaddvariableop_3_resource: (
assignaddvariableop_4_resource:	 (
assignaddvariableop_5_resource:	 
identity

identity_1

identity_2��AssignAddVariableOp�AssignAddVariableOp_1�AssignAddVariableOp_2�AssignAddVariableOp_3�AssignAddVariableOp_4�AssignAddVariableOp_5�StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallbatch_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8� *,
f'R%
#__inference_predict_on_batch_506685j
%sparse_categorical_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3j
%sparse_categorical_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
#sparse_categorical_crossentropy/subSub.sparse_categorical_crossentropy/sub/x:output:0.sparse_categorical_crossentropy/Const:output:0*
T0*
_output_shapes
: �
5sparse_categorical_crossentropy/clip_by_value/MinimumMinimum StatefulPartitionedCall:output:0'sparse_categorical_crossentropy/sub:z:0*
T0*'
_output_shapes
:���������
�
-sparse_categorical_crossentropy/clip_by_valueMaximum9sparse_categorical_crossentropy/clip_by_value/Minimum:z:0.sparse_categorical_crossentropy/Const:output:0*
T0*'
_output_shapes
:���������
�
#sparse_categorical_crossentropy/LogLog1sparse_categorical_crossentropy/clip_by_value:z:0*
T0*'
_output_shapes
:���������
�
%sparse_categorical_crossentropy/ShapeShape'sparse_categorical_crossentropy/Log:y:0*
T0*
_output_shapes
::���
Isparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ShapeShapebatch_input_1*
T0	*
_output_shapes
::���
gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits'sparse_categorical_crossentropy/Log:y:0batch_input_1*
T0*6
_output_shapes$
":���������:���������
x
3sparse_categorical_crossentropy/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
1sparse_categorical_crossentropy/weighted_loss/MulMulnsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:loss:0<sparse_categorical_crossentropy/weighted_loss/Const:output:0*
T0*#
_output_shapes
:���������
5sparse_categorical_crossentropy/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
1sparse_categorical_crossentropy/weighted_loss/SumSum5sparse_categorical_crossentropy/weighted_loss/Mul:z:0>sparse_categorical_crossentropy/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: �
:sparse_categorical_crossentropy/weighted_loss/num_elementsSize5sparse_categorical_crossentropy/weighted_loss/Mul:z:0*
T0*
_output_shapes
: �
?sparse_categorical_crossentropy/weighted_loss/num_elements/CastCastCsparse_categorical_crossentropy/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: t
2sparse_categorical_crossentropy/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : {
9sparse_categorical_crossentropy/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : {
9sparse_categorical_crossentropy/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
3sparse_categorical_crossentropy/weighted_loss/rangeRangeBsparse_categorical_crossentropy/weighted_loss/range/start:output:0;sparse_categorical_crossentropy/weighted_loss/Rank:output:0Bsparse_categorical_crossentropy/weighted_loss/range/delta:output:0*
_output_shapes
: �
3sparse_categorical_crossentropy/weighted_loss/Sum_1Sum:sparse_categorical_crossentropy/weighted_loss/Sum:output:0<sparse_categorical_crossentropy/weighted_loss/range:output:0*
T0*
_output_shapes
: �
3sparse_categorical_crossentropy/weighted_loss/valueDivNoNan<sparse_categorical_crossentropy/weighted_loss/Sum_1:output:0Csparse_categorical_crossentropy/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: X
CastCastbatch_input_1*

DstT0*

SrcT0	*#
_output_shapes
:���������K
ShapeShapeCast:y:0*
T0*
_output_shapes
::��[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������{
ArgMaxArgMax StatefulPartitionedCall:output:0ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������\
Cast_1CastArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������R
EqualEqualCast:y:0
Cast_1:y:0*
T0*#
_output_shapes
:���������V
Cast_2Cast	Equal:z:0*

DstT0*

SrcT0
*#
_output_shapes
:���������O
ConstConst*
_output_shapes
:*
dtype0*
valueB: o
SumSum
Cast_2:y:0Const:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: �
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype09
SizeSize
Cast_2:y:0*
T0*
_output_shapes
: M
Cast_3CastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: �
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resource
Cast_3:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0e
Shape_1Shape StatefulPartitionedCall:output:0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape_1:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
'sparse_categorical_crossentropy_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *���3l
'sparse_categorical_crossentropy_1/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
%sparse_categorical_crossentropy_1/subSub0sparse_categorical_crossentropy_1/sub/x:output:00sparse_categorical_crossentropy_1/Const:output:0*
T0*
_output_shapes
: �
7sparse_categorical_crossentropy_1/clip_by_value/MinimumMinimum StatefulPartitionedCall:output:0)sparse_categorical_crossentropy_1/sub:z:0*
T0*'
_output_shapes
:���������
�
/sparse_categorical_crossentropy_1/clip_by_valueMaximum;sparse_categorical_crossentropy_1/clip_by_value/Minimum:z:00sparse_categorical_crossentropy_1/Const:output:0*
T0*'
_output_shapes
:���������
�
%sparse_categorical_crossentropy_1/LogLog3sparse_categorical_crossentropy_1/clip_by_value:z:0*
T0*'
_output_shapes
:���������
�
'sparse_categorical_crossentropy_1/ShapeShape)sparse_categorical_crossentropy_1/Log:y:0*
T0*
_output_shapes
::���
Ksparse_categorical_crossentropy_1/SparseSoftmaxCrossEntropyWithLogits/ShapeShapebatch_input_1*
T0	*
_output_shapes
::���
isparse_categorical_crossentropy_1/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits)sparse_categorical_crossentropy_1/Log:y:0batch_input_1*
T0*6
_output_shapes$
":���������:���������
z
5sparse_categorical_crossentropy_1/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3sparse_categorical_crossentropy_1/weighted_loss/MulMulpsparse_categorical_crossentropy_1/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:loss:0>sparse_categorical_crossentropy_1/weighted_loss/Const:output:0*
T0*#
_output_shapes
:����������
7sparse_categorical_crossentropy_1/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
3sparse_categorical_crossentropy_1/weighted_loss/SumSum7sparse_categorical_crossentropy_1/weighted_loss/Mul:z:0@sparse_categorical_crossentropy_1/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: �
<sparse_categorical_crossentropy_1/weighted_loss/num_elementsSize7sparse_categorical_crossentropy_1/weighted_loss/Mul:z:0*
T0*
_output_shapes
: �
Asparse_categorical_crossentropy_1/weighted_loss/num_elements/CastCastEsparse_categorical_crossentropy_1/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: v
4sparse_categorical_crossentropy_1/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : }
;sparse_categorical_crossentropy_1/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : }
;sparse_categorical_crossentropy_1/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
5sparse_categorical_crossentropy_1/weighted_loss/rangeRangeDsparse_categorical_crossentropy_1/weighted_loss/range/start:output:0=sparse_categorical_crossentropy_1/weighted_loss/Rank:output:0Dsparse_categorical_crossentropy_1/weighted_loss/range/delta:output:0*
_output_shapes
: �
5sparse_categorical_crossentropy_1/weighted_loss/Sum_1Sum<sparse_categorical_crossentropy_1/weighted_loss/Sum:output:0>sparse_categorical_crossentropy_1/weighted_loss/range:output:0*
T0*
_output_shapes
: �
5sparse_categorical_crossentropy_1/weighted_loss/valueDivNoNan>sparse_categorical_crossentropy_1/weighted_loss/Sum_1:output:0Esparse_categorical_crossentropy_1/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: V
Cast_4Caststrided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: r
MulMul9sparse_categorical_crossentropy_1/weighted_loss/value:z:0
Cast_4:y:0*
T0*
_output_shapes
: F
RankConst*
_output_shapes
: *
dtype0*
value	B : M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :c
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
: n
Sum_1SumMul:z:0range:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
: �
AssignAddVariableOp_2AssignAddVariableOpassignaddvariableop_2_resourceSum_1:output:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0H
Rank_1Const*
_output_shapes
: *
dtype0*
value	B : O
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*
_output_shapes
: K
Sum_2Sum
Cast_4:y:0range_1:output:0*
T0*
_output_shapes
: �
AssignAddVariableOp_3AssignAddVariableOpassignaddvariableop_3_resourceSum_2:output:0^AssignAddVariableOp_2*
_output_shapes
 *
dtype0R
Shape_2Shapebatch_input_1*
T0	*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_5Caststrided_slice_1:output:0*

DstT0	*

SrcT0*
_output_shapes
: H
Rank_2Const*
_output_shapes
: *
dtype0*
value	B : O
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_2Rangerange_2/start:output:0Rank_2:output:0range_2/delta:output:0*
_output_shapes
: s
Sum_3Sum
Cast_5:y:0range_2:output:0*
T0	*&
 _has_manual_control_dependencies(*
_output_shapes
: 
AssignAddVariableOp_4AssignAddVariableOpassignaddvariableop_4_resourceSum_3:output:0*
_output_shapes
 *
dtype0	J
Cast_6/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_6CastCast_6/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: H
Rank_3Const*
_output_shapes
: *
dtype0*
value	B : O
range_3/startConst*
_output_shapes
: *
dtype0*
value	B : O
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :k
range_3Rangerange_3/start:output:0Rank_3:output:0range_3/delta:output:0*
_output_shapes
: s
Sum_4Sum
Cast_6:y:0range_3:output:0*
T0	*&
 _has_manual_control_dependencies(*
_output_shapes
: 
AssignAddVariableOp_5AssignAddVariableOpassignaddvariableop_5_resourceSum_4:output:0*
_output_shapes
 *
dtype0	P
Shape_3Shapebatch_input*
T0	*
_output_shapes
::��_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape_3:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
IdentityIdentity7sparse_categorical_crossentropy/weighted_loss/value:z:0^NoOp*
T0*
_output_shapes
: q

Identity_1Identity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
X

Identity_2Identitystrided_slice_2:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^AssignAddVariableOp_2^AssignAddVariableOp_3^AssignAddVariableOp_4^AssignAddVariableOp_5^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:���������  :���������: : : : : : : : : : : : : : : : 2.
AssignAddVariableOp_1AssignAddVariableOp_12.
AssignAddVariableOp_2AssignAddVariableOp_22.
AssignAddVariableOp_3AssignAddVariableOp_32.
AssignAddVariableOp_4AssignAddVariableOp_42.
AssignAddVariableOp_5AssignAddVariableOp_52*
AssignAddVariableOpAssignAddVariableOp22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:&"
 
_user_specified_name506704:&
"
 
_user_specified_name506702:&	"
 
_user_specified_name506700:&"
 
_user_specified_name506698:&"
 
_user_specified_name506696:&"
 
_user_specified_name506694:&"
 
_user_specified_name506692:&"
 
_user_specified_name506690:&"
 
_user_specified_name506688:&"
 
_user_specified_name506686:PL
#
_output_shapes
:���������
%
_user_specified_namebatch_input:\ X
/
_output_shapes
:���������  
%
_user_specified_namebatch_input
�
�
#__inference_flattened_output_506892

args_0	
args_0_1	!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5:	�@
	unknown_6:@
	unknown_7:@

	unknown_8:

	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13:	 

unknown_14:	 
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
: :���������
: *,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2*0,1J 8� *(
f#R!
__inference_forward_pass_506811^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������
`

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*
_output_shapes
: <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:���������  :���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name506884:&"
 
_user_specified_name506882:&"
 
_user_specified_name506880:&"
 
_user_specified_name506878:&"
 
_user_specified_name506876:&"
 
_user_specified_name506874:&"
 
_user_specified_name506872:&
"
 
_user_specified_name506870:&	"
 
_user_specified_name506868:&"
 
_user_specified_name506866:&"
 
_user_specified_name506864:&"
 
_user_specified_name506862:&"
 
_user_specified_name506860:&"
 
_user_specified_name506858:&"
 
_user_specified_name506856:&"
 
_user_specified_name506854:KG
#
_output_shapes
:���������
 
_user_specified_nameargs_0:W S
/
_output_shapes
:���������  
 
_user_specified_nameargs_0"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
A
args_07
serving_default_args_0:0	���������  <
output_00
StatefulPartitionedCall:0���������
tensorflow/serving/predict:�"
�
tff_trainable_variables
tff_non_trainable_variables
tff_local_variables
#forward_pass_training_type_spec
$ forward_pass_inference_type_spec
'#predict_on_batch_training_type_spec
($predict_on_batch_inference_type_spec
 serialized_metric_finalizers
	serialized_input_spec

flat_forward_pass_inference
flat_forward_pass_training
predict_on_batch_inference
predict_on_batch_training
$ report_local_unfinalized_metrics
reset_metrics

signatures"
_generic_user_object
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
 5"
trackable_list_wrapper
: 2Variable
: 2Variable
: 2Variable
: 2Variable
n
!sparse_categorical_accuracy
"loss
#num_examples
$num_batches"
trackable_dict_wrapper
: 2Variable
�B�
#__inference_flattened_output_506892args_0args_0_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_flattened_output_506623args_0args_0_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_flattened_output_506998args_0"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_flattened_output_506945args_0"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_report_local_unfinalized_metrics_507027"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
 __inference_reset_metrics_507164"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
,
%serving_default"
signature_map
):' 2conv2d_3/kernel
: 2conv2d_3/bias
):'  2conv2d_4/kernel
: 2conv2d_4/bias
):'  2conv2d_5/kernel
: 2conv2d_5/bias
!:	�@2dense_2/kernel
:@2dense_2/bias
 :@
2dense_3/kernel
:
2dense_3/bias
:  (2total
:  (2count
:  (2total
:  (2count
:	  (2total
:	  (2total
: 2Variable
: 2Variable
: 2Variable
: 2Variable
�B�
$__inference_signature_wrapper_507190args_0"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jargs_0
kwonlydefaults
 
annotations� *
 �
#__inference_flattened_output_506623� h�e
^�[
Y�V
/
x*�'
args_0_x���������  	
#
y�
args_0_y���������	
� "M�J
�
tensor_0 
"�
tensor_1���������

�
tensor_2 �
#__inference_flattened_output_506892� h�e
^�[
Y�V
/
x*�'
args_0_x���������  	
#
y�
args_0_y���������	
� "M�J
�
tensor_0 
"�
tensor_1���������

�
tensor_2 �
#__inference_flattened_output_506945n
7�4
-�*
(�%
args_0���������  	
� "'�$
"�
tensor_0���������
�
#__inference_flattened_output_506998n
7�4
-�*
(�%
args_0���������  	
� "'�$
"�
tensor_0���������
�
3__inference_report_local_unfinalized_metrics_507027� �

� 
� "���
-
loss%�"
�
loss_0 
�
loss_1 
*
num_batches�
�
num_batches_0 	
,
num_examples�
�
num_examples_0 	
r
sparse_categorical_accuracyS�P
&�#
sparse_categorical_accuracy_0 
&�#
sparse_categorical_accuracy_1 ?
 __inference_reset_metrics_507164 �

� 
� "
 �
$__inference_signature_wrapper_507190�
A�>
� 
7�4
2
args_0(�%
args_0���������  	"3�0
.
output_0"�
output_0���������
