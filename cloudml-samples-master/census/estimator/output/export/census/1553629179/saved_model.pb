ёЙ
щ3Й3
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
2	
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
З
AsString

input"T

output"
Ttype:

2	
"
	precisionintџџџџџџџџџ"

scientificbool( "
shortestbool( "
widthintџџџџџџџџџ"
fillstring 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
S
	Bucketize

input"T

output"
Ttype:
2	"

boundarieslist(float)
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
Ё
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
b
InitializeTableV2
table_handle
keys"Tkey
values"Tval"
Tkeytype"
Tvaltype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
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

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	


OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisintџџџџџџџџџ"	
Ttype"
TItype0	:
2	
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
@
ReadVariableOp
resource
value"dtype"
dtypetype
D
Relu
features"T
activations"T"
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
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
0
Sigmoid
x"T
y"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
№
SparseCross
indices	*N
values2sparse_types
shapes	*N
dense_inputs2dense_types
output_indices	
output_values"out_type
output_shape	"

Nint("
hashed_outputbool"
num_bucketsint("
hash_keyint"$
sparse_types
list(type)(:
2	"#
dense_types
list(type)(:
2	"
out_typetype:
2	"
internal_typetype:
2	
З
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	

SparseSegmentSum	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2	"
Tidxtype0:
2	
М
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized

s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.10.02v1.10.0-rc1-19-g656e7a2b348к


global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
dtype0	*
_output_shapes
: *
shape: *
_class
loc:@global_step

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
f
PlaceholderPlaceholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_3Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
h
Placeholder_4Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ
h
Placeholder_5Placeholder*
shape:џџџџџџџџџ*
dtype0*#
_output_shapes
:џџџџџџџџџ
h
Placeholder_6Placeholder*
dtype0*#
_output_shapes
:џџџџџџџџџ*
shape:џџџџџџџџџ

Jdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
п
Fdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDims
ExpandDimsPlaceholderJdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ

Zdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
Ж
Tdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/NotEqualNotEqualFdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDimsZdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:џџџџџџџџџ
л
Sdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/indicesWhereTdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/NotEqual*'
_output_shapes
:џџџџџџџџџ
П
Rdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/valuesGatherNdFdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDimsSdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/indices*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Tindices0	
н
Wdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/dense_shapeShapeFdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDims*
_output_shapes
:*
T0*
out_type0	

Odnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/ConstConst*
valueBB FB M*
dtype0*
_output_shapes
:

Ndnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/SizeConst*
dtype0*
_output_shapes
: *
value	B :

Udnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Udnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
ђ
Odnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/rangeRangeUdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/range/startNdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/SizeUdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/range/delta*
_output_shapes
:
о
Qdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/ToInt64CastOdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/range*

SrcT0*
_output_shapes
:*

DstT0	

Tdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_tableHashTableV2*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
Ѕ
Zdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/ConstConst*
valueB	 R
џџџџџџџџџ*
dtype0	*
_output_shapes
: 

_dnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/table_initInitializeTableV2Tdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_tableOdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/ConstQdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/ToInt64*

Tkey0*

Tval0	
Є
Mdnn/input_from_feature_columns/input_layer/gender_indicator/hash_table_LookupLookupTableFindV2Tdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_tableRdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/valuesZdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/Const*

Tout0	*#
_output_shapes
:џџџџџџџџџ*	
Tin0
Ђ
Wdnn/input_from_feature_columns/input_layer/gender_indicator/SparseToDense/default_valueConst*
valueB	 R
џџџџџџџџџ*
dtype0	*
_output_shapes
: 
ђ
Idnn/input_from_feature_columns/input_layer/gender_indicator/SparseToDenseSparseToDenseSdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/indicesWdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/dense_shapeMdnn/input_from_feature_columns/input_layer/gender_indicator/hash_table_LookupWdnn/input_from_feature_columns/input_layer/gender_indicator/SparseToDense/default_value*
Tindices0	*
T0	*'
_output_shapes
:џџџџџџџџџ

Idnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Kdnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    

Idnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 

Ldnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/on_valueConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Mdnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/off_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ж
Cdnn/input_from_feature_columns/input_layer/gender_indicator/one_hotOneHotIdnn/input_from_feature_columns/input_layer/gender_indicator/SparseToDenseIdnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/depthLdnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/on_valueMdnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/off_value*
T0*+
_output_shapes
:џџџџџџџџџ
Є
Qdnn/input_from_feature_columns/input_layer/gender_indicator/Sum/reduction_indicesConst*
valueB:
ўџџџџџџџџ*
dtype0*
_output_shapes
:

?dnn/input_from_feature_columns/input_layer/gender_indicator/SumSumCdnn/input_from_feature_columns/input_layer/gender_indicator/one_hotQdnn/input_from_feature_columns/input_layer/gender_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ
А
Adnn/input_from_feature_columns/input_layer/gender_indicator/ShapeShape?dnn/input_from_feature_columns/input_layer/gender_indicator/Sum*
T0*
_output_shapes
:

Odnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Qdnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Qdnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
й
Idnn/input_from_feature_columns/input_layer/gender_indicator/strided_sliceStridedSliceAdnn/input_from_feature_columns/input_layer/gender_indicator/ShapeOdnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stackQdnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stack_1Qdnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0

Kdnn/input_from_feature_columns/input_layer/gender_indicator/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 

Idnn/input_from_feature_columns/input_layer/gender_indicator/Reshape/shapePackIdnn/input_from_feature_columns/input_layer/gender_indicator/strided_sliceKdnn/input_from_feature_columns/input_layer/gender_indicator/Reshape/shape/1*
T0*
N*
_output_shapes
:

Cdnn/input_from_feature_columns/input_layer/gender_indicator/ReshapeReshape?dnn/input_from_feature_columns/input_layer/gender_indicator/SumIdnn/input_from_feature_columns/input_layer/gender_indicator/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0

Hdnn/input_from_feature_columns/input_layer/type_indicator/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
н
Ddnn/input_from_feature_columns/input_layer/type_indicator/ExpandDims
ExpandDimsPlaceholder_1Hdnn/input_from_feature_columns/input_layer/type_indicator/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*
T0

Xdnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
А
Rdnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/NotEqualNotEqualDdnn/input_from_feature_columns/input_layer/type_indicator/ExpandDimsXdnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/ignore_value/x*'
_output_shapes
:џџџџџџџџџ*
T0
з
Qdnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/indicesWhereRdnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/NotEqual*'
_output_shapes
:џџџџџџџџџ
Й
Pdnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/valuesGatherNdDdnn/input_from_feature_columns/input_layer/type_indicator/ExpandDimsQdnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ
й
Udnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/dense_shapeShapeDdnn/input_from_feature_columns/input_layer/type_indicator/ExpandDims*
_output_shapes
:*
T0*
out_type0	
І
Kdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/ConstConst*І
valueBB sarituri ca mingeaB sarituri ca mingea cu greutateB sarituri duble greutateB sarituri dubleB sarituri simpleB sarituri simple cu greutate*
dtype0*
_output_shapes
:

Jdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/SizeConst*
dtype0*
_output_shapes
: *
value	B :

Qdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/range/startConst*
dtype0*
_output_shapes
: *
value	B : 

Qdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
т
Kdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/rangeRangeQdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/range/startJdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/SizeQdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/range/delta*
_output_shapes
:
ж
Mdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/ToInt64CastKdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/range*
_output_shapes
:*

DstT0	*

SrcT0

Pdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/hash_tableHashTableV2*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
Ё
Vdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/hash_table/ConstConst*
valueB	 R
џџџџџџџџџ*
dtype0	*
_output_shapes
: 
і
[dnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/hash_table/table_initInitializeTableV2Pdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/hash_tableKdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/ConstMdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/ToInt64*

Tkey0*

Tval0	

Kdnn/input_from_feature_columns/input_layer/type_indicator/hash_table_LookupLookupTableFindV2Pdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/hash_tablePdnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/valuesVdnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/hash_table/Const*

Tout0	*#
_output_shapes
:џџџџџџџџџ*	
Tin0
 
Udnn/input_from_feature_columns/input_layer/type_indicator/SparseToDense/default_valueConst*
dtype0	*
_output_shapes
: *
valueB	 R
џџџџџџџџџ
ш
Gdnn/input_from_feature_columns/input_layer/type_indicator/SparseToDenseSparseToDenseQdnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/indicesUdnn/input_from_feature_columns/input_layer/type_indicator/to_sparse_input/dense_shapeKdnn/input_from_feature_columns/input_layer/type_indicator/hash_table_LookupUdnn/input_from_feature_columns/input_layer/type_indicator/SparseToDense/default_value*
Tindices0	*
T0	*'
_output_shapes
:џџџџџџџџџ

Gdnn/input_from_feature_columns/input_layer/type_indicator/one_hot/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

Idnn/input_from_feature_columns/input_layer/type_indicator/one_hot/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 

Gdnn/input_from_feature_columns/input_layer/type_indicator/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 

Jdnn/input_from_feature_columns/input_layer/type_indicator/one_hot/on_valueConst*
dtype0*
_output_shapes
: *
valueB
 *  ?

Kdnn/input_from_feature_columns/input_layer/type_indicator/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ќ
Adnn/input_from_feature_columns/input_layer/type_indicator/one_hotOneHotGdnn/input_from_feature_columns/input_layer/type_indicator/SparseToDenseGdnn/input_from_feature_columns/input_layer/type_indicator/one_hot/depthJdnn/input_from_feature_columns/input_layer/type_indicator/one_hot/on_valueKdnn/input_from_feature_columns/input_layer/type_indicator/one_hot/off_value*+
_output_shapes
:џџџџџџџџџ*
T0
Ђ
Odnn/input_from_feature_columns/input_layer/type_indicator/Sum/reduction_indicesConst*
valueB:
ўџџџџџџџџ*
dtype0*
_output_shapes
:

=dnn/input_from_feature_columns/input_layer/type_indicator/SumSumAdnn/input_from_feature_columns/input_layer/type_indicator/one_hotOdnn/input_from_feature_columns/input_layer/type_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
?dnn/input_from_feature_columns/input_layer/type_indicator/ShapeShape=dnn/input_from_feature_columns/input_layer/type_indicator/Sum*
T0*
_output_shapes
:

Mdnn/input_from_feature_columns/input_layer/type_indicator/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Odnn/input_from_feature_columns/input_layer/type_indicator/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Odnn/input_from_feature_columns/input_layer/type_indicator/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Я
Gdnn/input_from_feature_columns/input_layer/type_indicator/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/type_indicator/ShapeMdnn/input_from_feature_columns/input_layer/type_indicator/strided_slice/stackOdnn/input_from_feature_columns/input_layer/type_indicator/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/type_indicator/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 

Idnn/input_from_feature_columns/input_layer/type_indicator/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :

Gdnn/input_from_feature_columns/input_layer/type_indicator/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/type_indicator/strided_sliceIdnn/input_from_feature_columns/input_layer/type_indicator/Reshape/shape/1*
T0*
N*
_output_shapes
:

Adnn/input_from_feature_columns/input_layer/type_indicator/ReshapeReshape=dnn/input_from_feature_columns/input_layer/type_indicator/SumGdnn/input_from_feature_columns/input_layer/type_indicator/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
И
1dnn/input_from_feature_columns/input_layer/concatConcatV2Cdnn/input_from_feature_columns/input_layer/gender_indicator/ReshapeAdnn/input_from_feature_columns/input_layer/type_indicator/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Х
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB"   d   *
dtype0*
_output_shapes
:
З
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *я[qО*
dtype0*
_output_shapes
: 
З
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *я[q>*
dtype0*
_output_shapes
: 

Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:d

>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
Ќ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:d

:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:d
Ѓ
dnn/hiddenlayer_0/kernel/part_0
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:d*
shape
:d
ъ
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:d
Ў
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:d
Ў
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
valueBd*    *
dtype0*
_output_shapes
:d

dnn/hiddenlayer_0/bias/part_0
VariableV2*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:d*
shape:d
е
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:d
Є
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:d
s
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
T0*
_output_shapes

:d
Ё
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*'
_output_shapes
:џџџџџџџџџd
k
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
_output_shapes
:d*
T0

dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*'
_output_shapes
:џџџџџџџџџd
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџd
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*'
_output_shapes
:џџџџџџџџџd
x
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

SrcT0
*'
_output_shapes
:џџџџџџџџџd*

DstT0
h
dnn/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
p
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values
Ћ
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: 
Х
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB"d   F   *
dtype0*
_output_shapes
:
З
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB
 *H`@О*
dtype0*
_output_shapes
: 
З
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB
 *H`@>*
dtype0*
_output_shapes
: 

Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:dF

>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Ќ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:dF

:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:dF*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Ѓ
dnn/hiddenlayer_1/kernel/part_0
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:dF*
shape
:dF
ъ
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
_output_shapes

:dF*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Ў
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:dF
Ў
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:F*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
valueBF*    

dnn/hiddenlayer_1/bias/part_0
VariableV2*
dtype0*
_output_shapes
:F*
shape:F*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
е
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:F
Є
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:F
s
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
_output_shapes

:dF*
T0

dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:џџџџџџџџџF
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
_output_shapes
:F*
T0

dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*'
_output_shapes
:џџџџџџџџџF
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџF
]
dnn/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 

dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*'
_output_shapes
:џџџџџџџџџF
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*'
_output_shapes
:џџџџџџџџџF*

DstT0*

SrcT0

j
dnn/zero_fraction_1/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
v
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
­
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: 
Х
@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
valueB"F   0   *
dtype0*
_output_shapes
:
З
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
valueB
 *йчfО*
dtype0*
_output_shapes
: 
З
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
valueB
 *йчf>*
dtype0*
_output_shapes
: 

Hdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes

:F0

>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: 
Ќ
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

:F0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0

:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:F0
Ѓ
dnn/hiddenlayer_2/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:F0*
shape
:F0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
ъ
&dnn/hiddenlayer_2/kernel/part_0/AssignAssigndnn/hiddenlayer_2/kernel/part_0:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform*
_output_shapes

:F0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
Ў
$dnn/hiddenlayer_2/kernel/part_0/readIdentitydnn/hiddenlayer_2/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:F0
Ў
/dnn/hiddenlayer_2/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
valueB0*    *
dtype0*
_output_shapes
:0

dnn/hiddenlayer_2/bias/part_0
VariableV2*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
:0*
shape:0
е
$dnn/hiddenlayer_2/bias/part_0/AssignAssigndnn/hiddenlayer_2/bias/part_0/dnn/hiddenlayer_2/bias/part_0/Initializer/zeros*
_output_shapes
:0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0
Є
"dnn/hiddenlayer_2/bias/part_0/readIdentitydnn/hiddenlayer_2/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
:0
s
dnn/hiddenlayer_2/kernelIdentity$dnn/hiddenlayer_2/kernel/part_0/read*
_output_shapes

:F0*
T0

dnn/hiddenlayer_2/MatMulMatMuldnn/hiddenlayer_1/Reludnn/hiddenlayer_2/kernel*
T0*'
_output_shapes
:џџџџџџџџџ0
k
dnn/hiddenlayer_2/biasIdentity"dnn/hiddenlayer_2/bias/part_0/read*
_output_shapes
:0*
T0

dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/bias*
T0*'
_output_shapes
:џџџџџџџџџ0
k
dnn/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ0
]
dnn/zero_fraction_2/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    

dnn/zero_fraction_2/EqualEqualdnn/hiddenlayer_2/Reludnn/zero_fraction_2/zero*'
_output_shapes
:џџџџџџџџџ0*
T0
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*'
_output_shapes
:џџџџџџџџџ0*

DstT0*

SrcT0

j
dnn/zero_fraction_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
v
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values*
dtype0*
_output_shapes
: 
­
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_2/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_2/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tagdnn/hiddenlayer_2/Relu*
_output_shapes
: 
Х
@dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
valueB"0   "   *
dtype0*
_output_shapes
:
З
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
valueB
 *О*
dtype0*
_output_shapes
: 
З
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
valueB
 *>

Hdnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes

:0"

>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0
Ќ
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

:0"*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0

:dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:0"*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0
Ѓ
dnn/hiddenlayer_3/kernel/part_0
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes

:0"*
shape
:0"
ъ
&dnn/hiddenlayer_3/kernel/part_0/AssignAssigndnn/hiddenlayer_3/kernel/part_0:dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
_output_shapes

:0"
Ў
$dnn/hiddenlayer_3/kernel/part_0/readIdentitydnn/hiddenlayer_3/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
_output_shapes

:0"
Ў
/dnn/hiddenlayer_3/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0*
valueB"*    *
dtype0*
_output_shapes
:"

dnn/hiddenlayer_3/bias/part_0
VariableV2*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0*
dtype0*
_output_shapes
:"*
shape:"
е
$dnn/hiddenlayer_3/bias/part_0/AssignAssigndnn/hiddenlayer_3/bias/part_0/dnn/hiddenlayer_3/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0*
_output_shapes
:"
Є
"dnn/hiddenlayer_3/bias/part_0/readIdentitydnn/hiddenlayer_3/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0*
_output_shapes
:"
s
dnn/hiddenlayer_3/kernelIdentity$dnn/hiddenlayer_3/kernel/part_0/read*
T0*
_output_shapes

:0"

dnn/hiddenlayer_3/MatMulMatMuldnn/hiddenlayer_2/Reludnn/hiddenlayer_3/kernel*
T0*'
_output_shapes
:џџџџџџџџџ"
k
dnn/hiddenlayer_3/biasIdentity"dnn/hiddenlayer_3/bias/part_0/read*
T0*
_output_shapes
:"

dnn/hiddenlayer_3/BiasAddBiasAdddnn/hiddenlayer_3/MatMuldnn/hiddenlayer_3/bias*
T0*'
_output_shapes
:џџџџџџџџџ"
k
dnn/hiddenlayer_3/ReluReludnn/hiddenlayer_3/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ"
]
dnn/zero_fraction_3/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    

dnn/zero_fraction_3/EqualEqualdnn/hiddenlayer_3/Reludnn/zero_fraction_3/zero*'
_output_shapes
:џџџџџџџџџ"*
T0
|
dnn/zero_fraction_3/CastCastdnn/zero_fraction_3/Equal*

SrcT0
*'
_output_shapes
:џџџџџџџџџ"*

DstT0
j
dnn/zero_fraction_3/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
v
dnn/zero_fraction_3/MeanMeandnn/zero_fraction_3/Castdnn/zero_fraction_3/Const*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_3/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_3/fraction_of_zero_values*
dtype0*
_output_shapes
: 
­
-dnn/dnn/hiddenlayer_3/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_3/fraction_of_zero_values/tagsdnn/zero_fraction_3/Mean*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_3/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_3/activation*
dtype0*
_output_shapes
: 

 dnn/dnn/hiddenlayer_3/activationHistogramSummary$dnn/dnn/hiddenlayer_3/activation/tagdnn/hiddenlayer_3/Relu*
_output_shapes
: 
З
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB""      
Љ
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *їќгО*
dtype0*
_output_shapes
: 
Љ
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *їќг>*
dtype0*
_output_shapes
: 
№
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:"
ў
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 

7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:"

3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:"

dnn/logits/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:"*
shape
:"*+
_class!
loc:@dnn/logits/kernel/part_0
Ю
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:"

dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:"
 
(dnn/logits/bias/part_0/Initializer/zerosConst*)
_class
loc:@dnn/logits/bias/part_0*
valueB*    *
dtype0*
_output_shapes
:

dnn/logits/bias/part_0
VariableV2*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:*
shape:
Й
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
_output_shapes
:*
T0*)
_class
loc:@dnn/logits/bias/part_0

dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
_output_shapes

:"*
T0
x
dnn/logits/MatMulMatMuldnn/hiddenlayer_3/Reludnn/logits/kernel*
T0*'
_output_shapes
:џџџџџџџџџ
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
_output_shapes
:*
T0
s
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*'
_output_shapes
:џџџџџџџџџ
]
dnn/zero_fraction_4/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    

dnn/zero_fraction_4/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_4/zero*'
_output_shapes
:џџџџџџџџџ*
T0
|
dnn/zero_fraction_4/CastCastdnn/zero_fraction_4/Equal*'
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

j
dnn/zero_fraction_4/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
v
dnn/zero_fraction_4/MeanMeandnn/zero_fraction_4/Castdnn/zero_fraction_4/Const*
T0*
_output_shapes
: 

+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 

&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_4/Mean*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
x
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
о
Clinear/linear_model/age_bucketized/weights/part_0/Initializer/zerosConst*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
valueB*    *
dtype0*
_output_shapes

:

1linear/linear_model/age_bucketized/weights/part_0VarHandleOp*
shape
:*B
shared_name31linear/linear_model/age_bucketized/weights/part_0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
dtype0*
_output_shapes
: 
Г
Rlinear/linear_model/age_bucketized/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp1linear/linear_model/age_bucketized/weights/part_0*
_output_shapes
: 

8linear/linear_model/age_bucketized/weights/part_0/AssignAssignVariableOp1linear/linear_model/age_bucketized/weights/part_0Clinear/linear_model/age_bucketized/weights/part_0/Initializer/zeros*
dtype0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0
§
Elinear/linear_model/age_bucketized/weights/part_0/Read/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
dtype0*
_output_shapes

:

@linear/linear_model_1/linear_model/age_bucketized/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Э
<linear/linear_model_1/linear_model/age_bucketized/ExpandDims
ExpandDimsPlaceholder_2@linear/linear_model_1/linear_model/age_bucketized/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ
є
;linear/linear_model_1/linear_model/age_bucketized/Bucketize	Bucketize<linear/linear_model_1/linear_model/age_bucketized/ExpandDims*:

boundaries,
*"(  Р@   A   A  @A  `A  A  A   A  АA  РA*
T0*'
_output_shapes
:џџџџџџџџџ
Ђ
7linear/linear_model_1/linear_model/age_bucketized/ShapeShape;linear/linear_model_1/linear_model/age_bucketized/Bucketize*
_output_shapes
:*
T0

Elinear/linear_model_1/linear_model/age_bucketized/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Glinear/linear_model_1/linear_model/age_bucketized/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Glinear/linear_model_1/linear_model/age_bucketized/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ї
?linear/linear_model_1/linear_model/age_bucketized/strided_sliceStridedSlice7linear/linear_model_1/linear_model/age_bucketized/ShapeElinear/linear_model_1/linear_model/age_bucketized/strided_slice/stackGlinear/linear_model_1/linear_model/age_bucketized/strided_slice/stack_1Glinear/linear_model_1/linear_model/age_bucketized/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 

=linear/linear_model_1/linear_model/age_bucketized/range/startConst*
dtype0*
_output_shapes
: *
value	B : 

=linear/linear_model_1/linear_model/age_bucketized/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Є
7linear/linear_model_1/linear_model/age_bucketized/rangeRange=linear/linear_model_1/linear_model/age_bucketized/range/start?linear/linear_model_1/linear_model/age_bucketized/strided_slice=linear/linear_model_1/linear_model/age_bucketized/range/delta*#
_output_shapes
:џџџџџџџџџ

Blinear/linear_model_1/linear_model/age_bucketized/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B :
ћ
>linear/linear_model_1/linear_model/age_bucketized/ExpandDims_1
ExpandDims7linear/linear_model_1/linear_model/age_bucketized/rangeBlinear/linear_model_1/linear_model/age_bucketized/ExpandDims_1/dim*
T0*'
_output_shapes
:џџџџџџџџџ

@linear/linear_model_1/linear_model/age_bucketized/Tile/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:
ђ
6linear/linear_model_1/linear_model/age_bucketized/TileTile>linear/linear_model_1/linear_model/age_bucketized/ExpandDims_1@linear/linear_model_1/linear_model/age_bucketized/Tile/multiples*
T0*'
_output_shapes
:џџџџџџџџџ

?linear/linear_model_1/linear_model/age_bucketized/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
ы
9linear/linear_model_1/linear_model/age_bucketized/ReshapeReshape6linear/linear_model_1/linear_model/age_bucketized/Tile?linear/linear_model_1/linear_model/age_bucketized/Reshape/shape*
T0*#
_output_shapes
:џџџџџџџџџ

?linear/linear_model_1/linear_model/age_bucketized/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 

?linear/linear_model_1/linear_model/age_bucketized/range_1/limitConst*
value	B :*
dtype0*
_output_shapes
: 

?linear/linear_model_1/linear_model/age_bucketized/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ё
9linear/linear_model_1/linear_model/age_bucketized/range_1Range?linear/linear_model_1/linear_model/age_bucketized/range_1/start?linear/linear_model_1/linear_model/age_bucketized/range_1/limit?linear/linear_model_1/linear_model/age_bucketized/range_1/delta*
_output_shapes
:
Й
Blinear/linear_model_1/linear_model/age_bucketized/Tile_1/multiplesPack?linear/linear_model_1/linear_model/age_bucketized/strided_slice*
N*
_output_shapes
:*
T0
э
8linear/linear_model_1/linear_model/age_bucketized/Tile_1Tile9linear/linear_model_1/linear_model/age_bucketized/range_1Blinear/linear_model_1/linear_model/age_bucketized/Tile_1/multiples*
T0*#
_output_shapes
:џџџџџџџџџ

Alinear/linear_model_1/linear_model/age_bucketized/Reshape_1/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
є
;linear/linear_model_1/linear_model/age_bucketized/Reshape_1Reshape;linear/linear_model_1/linear_model/age_bucketized/BucketizeAlinear/linear_model_1/linear_model/age_bucketized/Reshape_1/shape*
T0*#
_output_shapes
:џџџџџџџџџ
y
7linear/linear_model_1/linear_model/age_bucketized/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
н
5linear/linear_model_1/linear_model/age_bucketized/mulMul7linear/linear_model_1/linear_model/age_bucketized/mul/x8linear/linear_model_1/linear_model/age_bucketized/Tile_1*
T0*#
_output_shapes
:џџџџџџџџџ
о
5linear/linear_model_1/linear_model/age_bucketized/addAdd;linear/linear_model_1/linear_model/age_bucketized/Reshape_15linear/linear_model_1/linear_model/age_bucketized/mul*
T0*#
_output_shapes
:џџџџџџџџџ
я
7linear/linear_model_1/linear_model/age_bucketized/stackPack9linear/linear_model_1/linear_model/age_bucketized/Reshape8linear/linear_model_1/linear_model/age_bucketized/Tile_1*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Ђ
@linear/linear_model_1/linear_model/age_bucketized/transpose/RankRank7linear/linear_model_1/linear_model/age_bucketized/stack*
T0*
_output_shapes
: 

Alinear/linear_model_1/linear_model/age_bucketized/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
ь
?linear/linear_model_1/linear_model/age_bucketized/transpose/subSub@linear/linear_model_1/linear_model/age_bucketized/transpose/RankAlinear/linear_model_1/linear_model/age_bucketized/transpose/sub/y*
T0*
_output_shapes
: 

Glinear/linear_model_1/linear_model/age_bucketized/transpose/Range/startConst*
dtype0*
_output_shapes
: *
value	B : 

Glinear/linear_model_1/linear_model/age_bucketized/transpose/Range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
У
Alinear/linear_model_1/linear_model/age_bucketized/transpose/RangeRangeGlinear/linear_model_1/linear_model/age_bucketized/transpose/Range/start@linear/linear_model_1/linear_model/age_bucketized/transpose/RankGlinear/linear_model_1/linear_model/age_bucketized/transpose/Range/delta*#
_output_shapes
:џџџџџџџџџ
њ
Alinear/linear_model_1/linear_model/age_bucketized/transpose/sub_1Sub?linear/linear_model_1/linear_model/age_bucketized/transpose/subAlinear/linear_model_1/linear_model/age_bucketized/transpose/Range*
T0*#
_output_shapes
:џџџџџџџџџ
і
;linear/linear_model_1/linear_model/age_bucketized/transpose	Transpose7linear/linear_model_1/linear_model/age_bucketized/stackAlinear/linear_model_1/linear_model/age_bucketized/transpose/sub_1*
T0*'
_output_shapes
:џџџџџџџџџ
П
9linear/linear_model_1/linear_model/age_bucketized/ToInt64Cast;linear/linear_model_1/linear_model/age_bucketized/transpose*

SrcT0*'
_output_shapes
:џџџџџџџџџ*

DstT0	
}
;linear/linear_model_1/linear_model/age_bucketized/stack_1/1Const*
value	B :*
dtype0*
_output_shapes
: 
э
9linear/linear_model_1/linear_model/age_bucketized/stack_1Pack?linear/linear_model_1/linear_model/age_bucketized/strided_slice;linear/linear_model_1/linear_model/age_bucketized/stack_1/1*
T0*
N*
_output_shapes
:
В
;linear/linear_model_1/linear_model/age_bucketized/ToInt64_1Cast9linear/linear_model_1/linear_model/age_bucketized/stack_1*

SrcT0*
_output_shapes
:*

DstT0	
З
>linear/linear_model_1/linear_model/age_bucketized/Shape_1/CastCast;linear/linear_model_1/linear_model/age_bucketized/ToInt64_1*
_output_shapes
:*

DstT0*

SrcT0	

Glinear/linear_model_1/linear_model/age_bucketized/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:

Ilinear/linear_model_1/linear_model/age_bucketized/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Ilinear/linear_model_1/linear_model/age_bucketized/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ж
Alinear/linear_model_1/linear_model/age_bucketized/strided_slice_1StridedSlice>linear/linear_model_1/linear_model/age_bucketized/Shape_1/CastGlinear/linear_model_1/linear_model/age_bucketized/strided_slice_1/stackIlinear/linear_model_1/linear_model/age_bucketized/strided_slice_1/stack_1Ilinear/linear_model_1/linear_model/age_bucketized/strided_slice_1/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

:linear/linear_model_1/linear_model/age_bucketized/Cast/x/1Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
э
8linear/linear_model_1/linear_model/age_bucketized/Cast/xPackAlinear/linear_model_1/linear_model/age_bucketized/strided_slice_1:linear/linear_model_1/linear_model/age_bucketized/Cast/x/1*
T0*
N*
_output_shapes
:
Ќ
6linear/linear_model_1/linear_model/age_bucketized/CastCast8linear/linear_model_1/linear_model/age_bucketized/Cast/x*
_output_shapes
:*

DstT0	*

SrcT0
Џ
?linear/linear_model_1/linear_model/age_bucketized/SparseReshapeSparseReshape9linear/linear_model_1/linear_model/age_bucketized/ToInt64;linear/linear_model_1/linear_model/age_bucketized/ToInt64_16linear/linear_model_1/linear_model/age_bucketized/Cast*-
_output_shapes
:џџџџџџџџџ:
Й
Hlinear/linear_model_1/linear_model/age_bucketized/SparseReshape/IdentityIdentity5linear/linear_model_1/linear_model/age_bucketized/add*
T0*#
_output_shapes
:џџџџџџџџџ
В
@linear/linear_model_1/linear_model/age_bucketized/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0*
dtype0*
_output_shapes

:

Jlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:

Ilinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
й
Dlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SliceSliceAlinear/linear_model_1/linear_model/age_bucketized/SparseReshape:1Jlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice/beginIlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:

Dlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/ConstConst*
valueB: *
dtype0*
_output_shapes
:
ј
Clinear/linear_model_1/linear_model/age_bucketized/weighted_sum/ProdProdDlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SliceDlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Const*
_output_shapes
: *
T0	

Olinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 

Llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
љ
Glinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2GatherV2Alinear/linear_model_1/linear_model/age_bucketized/SparseReshape:1Olinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2/indicesLlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2/axis*
Tindices0*
Tparams0	*
_output_shapes
: *
Taxis0

Elinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Cast/xPackClinear/linear_model_1/linear_model/age_bucketized/weighted_sum/ProdGlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
з
Llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseReshapeSparseReshape?linear/linear_model_1/linear_model/age_bucketized/SparseReshapeAlinear/linear_model_1/linear_model/age_bucketized/SparseReshape:1Elinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Cast/x*-
_output_shapes
:џџџџџџџџџ:
й
Ulinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseReshape/IdentityIdentityHlinear/linear_model_1/linear_model/age_bucketized/SparseReshape/Identity*
T0*#
_output_shapes
:џџџџџџџџџ

Mlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
Џ
Klinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GreaterEqualGreaterEqualUlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseReshape/IdentityMlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GreaterEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
У
Dlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/WhereWhereKlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GreaterEqual*'
_output_shapes
:џџџџџџџџџ

Llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:

Flinear/linear_model_1/linear_model/age_bucketized/weighted_sum/ReshapeReshapeDlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/WhereLlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Reshape/shape*#
_output_shapes
:џџџџџџџџџ*
T0	

Nlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

Ilinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2_1GatherV2Llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseReshapeFlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/ReshapeNlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2_1/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0	

Nlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

Ilinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2_2GatherV2Ulinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseReshape/IdentityFlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/ReshapeNlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2_2/axis*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	
Ш
Glinear/linear_model_1/linear_model/age_bucketized/weighted_sum/IdentityIdentityNlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:

Xlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B : *
dtype0*
_output_shapes
: 

flinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsIlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2_1Ilinear/linear_model_1/linear_model/age_bucketized/weighted_sum/GatherV2_2Glinear/linear_model_1/linear_model/age_bucketized/weighted_sum/IdentityXlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/Const*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
T0
Л
jlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Н
llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
Н
llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:

dlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceflinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsjlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stackllinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
Index0*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ

[linear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/CastCastdlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0

]linear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/UniqueUniquehlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup/axisConst*S
_classI
GEloc:@linear/linear_model_1/linear_model/age_bucketized/ReadVariableOp*
value	B : *
dtype0*
_output_shapes
: 
Ќ
glinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookupGatherV2@linear/linear_model_1/linear_model/age_bucketized/ReadVariableOp]linear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/Uniquellinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0*
Tparams0*S
_classI
GEloc:@linear/linear_model_1/linear_model/age_bucketized/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ*
Taxis0
У
Vlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparseSparseSegmentSumglinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup_linear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/Unique:1[linear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

Nlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ   
П
Hlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Reshape_1Reshapehlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Nlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ
Ъ
Dlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/ShapeShapeVlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse*
_output_shapes
:*
T0

Rlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:

Tlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

Tlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ш
Llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/strided_sliceStridedSliceDlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/ShapeRlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/strided_slice/stackTlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/strided_slice/stack_1Tlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0

Flinear/linear_model_1/linear_model/age_bucketized/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 

Dlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/stackPackFlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/stack/0Llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:

Clinear/linear_model_1/linear_model/age_bucketized/weighted_sum/TileTileHlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Reshape_1Dlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/stack*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

р
Ilinear/linear_model_1/linear_model/age_bucketized/weighted_sum/zeros_like	ZerosLikeVlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse*'
_output_shapes
:џџџџџџџџџ*
T0
т
>linear/linear_model_1/linear_model/age_bucketized/weighted_sumSelectClinear/linear_model_1/linear_model/age_bucketized/weighted_sum/TileIlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/zeros_likeVlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
Elinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Cast_1CastAlinear/linear_model_1/linear_model/age_bucketized/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0

Llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 

Klinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
у
Flinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_1SliceElinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Cast_1Llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_1/beginKlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
Д
Flinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Shape_1Shape>linear/linear_model_1/linear_model/age_bucketized/weighted_sum*
_output_shapes
:*
T0

Llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:

Klinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ф
Flinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_2SliceFlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Shape_1Llinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_2/beginKlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:

Jlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
л
Elinear/linear_model_1/linear_model/age_bucketized/weighted_sum/concatConcatV2Flinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_1Flinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Slice_2Jlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:

Hlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Reshape_2Reshape>linear/linear_model_1/linear_model/age_bucketized/weighted_sumElinear/linear_model_1/linear_model/age_bucketized/weighted_sum/concat*'
_output_shapes
:џџџџџџџџџ*
T0
Ю
;linear/linear_model/gender/weights/part_0/Initializer/zerosConst*
dtype0*
_output_shapes

:*<
_class2
0.loc:@linear/linear_model/gender/weights/part_0*
valueB*    
ь
)linear/linear_model/gender/weights/part_0VarHandleOp*
shape
:*:
shared_name+)linear/linear_model/gender/weights/part_0*<
_class2
0.loc:@linear/linear_model/gender/weights/part_0*
dtype0*
_output_shapes
: 
Ѓ
Jlinear/linear_model/gender/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp)linear/linear_model/gender/weights/part_0*
_output_shapes
: 
ї
0linear/linear_model/gender/weights/part_0/AssignAssignVariableOp)linear/linear_model/gender/weights/part_0;linear/linear_model/gender/weights/part_0/Initializer/zeros*<
_class2
0.loc:@linear/linear_model/gender/weights/part_0*
dtype0
х
=linear/linear_model/gender/weights/part_0/Read/ReadVariableOpReadVariableOp)linear/linear_model/gender/weights/part_0*<
_class2
0.loc:@linear/linear_model/gender/weights/part_0*
dtype0*
_output_shapes

:

8linear/linear_model_1/linear_model/gender/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Л
4linear/linear_model_1/linear_model/gender/ExpandDims
ExpandDimsPlaceholder8linear/linear_model_1/linear_model/gender/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ

Hlinear/linear_model_1/linear_model/gender/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 

Blinear/linear_model_1/linear_model/gender/to_sparse_input/NotEqualNotEqual4linear/linear_model_1/linear_model/gender/ExpandDimsHlinear/linear_model_1/linear_model/gender/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:џџџџџџџџџ
З
Alinear/linear_model_1/linear_model/gender/to_sparse_input/indicesWhereBlinear/linear_model_1/linear_model/gender/to_sparse_input/NotEqual*'
_output_shapes
:џџџџџџџџџ

@linear/linear_model_1/linear_model/gender/to_sparse_input/valuesGatherNd4linear/linear_model_1/linear_model/gender/ExpandDimsAlinear/linear_model_1/linear_model/gender/to_sparse_input/indices*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Tindices0	
Й
Elinear/linear_model_1/linear_model/gender/to_sparse_input/dense_shapeShape4linear/linear_model_1/linear_model/gender/ExpandDims*
T0*
out_type0	*
_output_shapes
:

=linear/linear_model_1/linear_model/gender/gender_lookup/ConstConst*
valueBB FB M*
dtype0*
_output_shapes
:
~
<linear/linear_model_1/linear_model/gender/gender_lookup/SizeConst*
value	B :*
dtype0*
_output_shapes
: 

Clinear/linear_model_1/linear_model/gender/gender_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 

Clinear/linear_model_1/linear_model/gender/gender_lookup/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
Њ
=linear/linear_model_1/linear_model/gender/gender_lookup/rangeRangeClinear/linear_model_1/linear_model/gender/gender_lookup/range/start<linear/linear_model_1/linear_model/gender/gender_lookup/SizeClinear/linear_model_1/linear_model/gender/gender_lookup/range/delta*
_output_shapes
:
К
?linear/linear_model_1/linear_model/gender/gender_lookup/ToInt64Cast=linear/linear_model_1/linear_model/gender/gender_lookup/range*

SrcT0*
_output_shapes
:*

DstT0	

Blinear/linear_model_1/linear_model/gender/gender_lookup/hash_tableHashTableV2*
value_dtype0	*
	key_dtype0*
_output_shapes
: 

Hlinear/linear_model_1/linear_model/gender/gender_lookup/hash_table/ConstConst*
valueB	 R
џџџџџџџџџ*
dtype0	*
_output_shapes
: 
О
Mlinear/linear_model_1/linear_model/gender/gender_lookup/hash_table/table_initInitializeTableV2Blinear/linear_model_1/linear_model/gender/gender_lookup/hash_table=linear/linear_model_1/linear_model/gender/gender_lookup/Const?linear/linear_model_1/linear_model/gender/gender_lookup/ToInt64*

Tkey0*

Tval0	
м
;linear/linear_model_1/linear_model/gender/hash_table_LookupLookupTableFindV2Blinear/linear_model_1/linear_model/gender/gender_lookup/hash_table@linear/linear_model_1/linear_model/gender/to_sparse_input/valuesHlinear/linear_model_1/linear_model/gender/gender_lookup/hash_table/Const*#
_output_shapes
:џџџџџџџџџ*	
Tin0*

Tout0	
З
4linear/linear_model_1/linear_model/gender/Shape/CastCastElinear/linear_model_1/linear_model/gender/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0

=linear/linear_model_1/linear_model/gender/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

?linear/linear_model_1/linear_model/gender/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

?linear/linear_model_1/linear_model/gender/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:

7linear/linear_model_1/linear_model/gender/strided_sliceStridedSlice4linear/linear_model_1/linear_model/gender/Shape/Cast=linear/linear_model_1/linear_model/gender/strided_slice/stack?linear/linear_model_1/linear_model/gender/strided_slice/stack_1?linear/linear_model_1/linear_model/gender/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
}
2linear/linear_model_1/linear_model/gender/Cast/x/1Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
г
0linear/linear_model_1/linear_model/gender/Cast/xPack7linear/linear_model_1/linear_model/gender/strided_slice2linear/linear_model_1/linear_model/gender/Cast/x/1*
T0*
N*
_output_shapes
:

.linear/linear_model_1/linear_model/gender/CastCast0linear/linear_model_1/linear_model/gender/Cast/x*

SrcT0*
_output_shapes
:*

DstT0	
Б
7linear/linear_model_1/linear_model/gender/SparseReshapeSparseReshapeAlinear/linear_model_1/linear_model/gender/to_sparse_input/indicesElinear/linear_model_1/linear_model/gender/to_sparse_input/dense_shape.linear/linear_model_1/linear_model/gender/Cast*-
_output_shapes
:џџџџџџџџџ:
З
@linear/linear_model_1/linear_model/gender/SparseReshape/IdentityIdentity;linear/linear_model_1/linear_model/gender/hash_table_Lookup*
T0	*#
_output_shapes
:џџџџџџџџџ
Ђ
8linear/linear_model_1/linear_model/gender/ReadVariableOpReadVariableOp)linear/linear_model/gender/weights/part_0*
dtype0*
_output_shapes

:

Blinear/linear_model_1/linear_model/gender/weighted_sum/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 

Alinear/linear_model_1/linear_model/gender/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Й
<linear/linear_model_1/linear_model/gender/weighted_sum/SliceSlice9linear/linear_model_1/linear_model/gender/SparseReshape:1Blinear/linear_model_1/linear_model/gender/weighted_sum/Slice/beginAlinear/linear_model_1/linear_model/gender/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:

<linear/linear_model_1/linear_model/gender/weighted_sum/ConstConst*
valueB: *
dtype0*
_output_shapes
:
р
;linear/linear_model_1/linear_model/gender/weighted_sum/ProdProd<linear/linear_model_1/linear_model/gender/weighted_sum/Slice<linear/linear_model_1/linear_model/gender/weighted_sum/Const*
T0	*
_output_shapes
: 

Glinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 

Dlinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
й
?linear/linear_model_1/linear_model/gender/weighted_sum/GatherV2GatherV29linear/linear_model_1/linear_model/gender/SparseReshape:1Glinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2/indicesDlinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
ё
=linear/linear_model_1/linear_model/gender/weighted_sum/Cast/xPack;linear/linear_model_1/linear_model/gender/weighted_sum/Prod?linear/linear_model_1/linear_model/gender/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
З
Dlinear/linear_model_1/linear_model/gender/weighted_sum/SparseReshapeSparseReshape7linear/linear_model_1/linear_model/gender/SparseReshape9linear/linear_model_1/linear_model/gender/SparseReshape:1=linear/linear_model_1/linear_model/gender/weighted_sum/Cast/x*-
_output_shapes
:џџџџџџџџџ:
Щ
Mlinear/linear_model_1/linear_model/gender/weighted_sum/SparseReshape/IdentityIdentity@linear/linear_model_1/linear_model/gender/SparseReshape/Identity*
T0	*#
_output_shapes
:џџџџџџџџџ

Elinear/linear_model_1/linear_model/gender/weighted_sum/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 

Clinear/linear_model_1/linear_model/gender/weighted_sum/GreaterEqualGreaterEqualMlinear/linear_model_1/linear_model/gender/weighted_sum/SparseReshape/IdentityElinear/linear_model_1/linear_model/gender/weighted_sum/GreaterEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0	
Г
<linear/linear_model_1/linear_model/gender/weighted_sum/WhereWhereClinear/linear_model_1/linear_model/gender/weighted_sum/GreaterEqual*'
_output_shapes
:џџџџџџџџџ

Dlinear/linear_model_1/linear_model/gender/weighted_sum/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ћ
>linear/linear_model_1/linear_model/gender/weighted_sum/ReshapeReshape<linear/linear_model_1/linear_model/gender/weighted_sum/WhereDlinear/linear_model_1/linear_model/gender/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ

Flinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
№
Alinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2_1GatherV2Dlinear/linear_model_1/linear_model/gender/weighted_sum/SparseReshape>linear/linear_model_1/linear_model/gender/weighted_sum/ReshapeFlinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2_1/axis*
Tindices0	*
Tparams0	*'
_output_shapes
:џџџџџџџџџ*
Taxis0

Flinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ѕ
Alinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2_2GatherV2Mlinear/linear_model_1/linear_model/gender/weighted_sum/SparseReshape/Identity>linear/linear_model_1/linear_model/gender/weighted_sum/ReshapeFlinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:џџџџџџџџџ*
Taxis0
И
?linear/linear_model_1/linear_model/gender/weighted_sum/IdentityIdentityFlinear/linear_model_1/linear_model/gender/weighted_sum/SparseReshape:1*
_output_shapes
:*
T0	

Plinear/linear_model_1/linear_model/gender/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
э
^linear/linear_model_1/linear_model/gender/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsAlinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2_1Alinear/linear_model_1/linear_model/gender/weighted_sum/GatherV2_2?linear/linear_model_1/linear_model/gender/weighted_sum/IdentityPlinear/linear_model_1/linear_model/gender/weighted_sum/SparseFillEmptyRows/Const*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
T0	
Г
blinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Е
dlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
Е
dlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ё
\linear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlice^linear/linear_model_1/linear_model/gender/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsblinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stackdlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1dlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*#
_output_shapes
:џџџџџџџџџ*
Index0*
T0	*
shrink_axis_mask*

begin_mask
і
Slinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/CastCast\linear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
ў
Ulinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/UniqueUnique`linear/linear_model_1/linear_model/gender/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0	
ѓ
dlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/embedding_lookup/axisConst*K
_classA
?=loc:@linear/linear_model_1/linear_model/gender/ReadVariableOp*
value	B : *
dtype0*
_output_shapes
: 

_linear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/embedding_lookupGatherV28linear/linear_model_1/linear_model/gender/ReadVariableOpUlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/Uniquedlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/embedding_lookup/axis*
Tparams0*K
_classA
?=loc:@linear/linear_model_1/linear_model/gender/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	
Ѓ
Nlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparseSparseSegmentSum_linear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/embedding_lookupWlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/Unique:1Slinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

Flinear/linear_model_1/linear_model/gender/weighted_sum/Reshape_1/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
Ї
@linear/linear_model_1/linear_model/gender/weighted_sum/Reshape_1Reshape`linear/linear_model_1/linear_model/gender/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Flinear/linear_model_1/linear_model/gender/weighted_sum/Reshape_1/shape*'
_output_shapes
:џџџџџџџџџ*
T0

К
<linear/linear_model_1/linear_model/gender/weighted_sum/ShapeShapeNlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:

Jlinear/linear_model_1/linear_model/gender/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:

Llinear/linear_model_1/linear_model/gender/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Llinear/linear_model_1/linear_model/gender/weighted_sum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Р
Dlinear/linear_model_1/linear_model/gender/weighted_sum/strided_sliceStridedSlice<linear/linear_model_1/linear_model/gender/weighted_sum/ShapeJlinear/linear_model_1/linear_model/gender/weighted_sum/strided_slice/stackLlinear/linear_model_1/linear_model/gender/weighted_sum/strided_slice/stack_1Llinear/linear_model_1/linear_model/gender/weighted_sum/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask

>linear/linear_model_1/linear_model/gender/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
ј
<linear/linear_model_1/linear_model/gender/weighted_sum/stackPack>linear/linear_model_1/linear_model/gender/weighted_sum/stack/0Dlinear/linear_model_1/linear_model/gender/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:
ў
;linear/linear_model_1/linear_model/gender/weighted_sum/TileTile@linear/linear_model_1/linear_model/gender/weighted_sum/Reshape_1<linear/linear_model_1/linear_model/gender/weighted_sum/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
а
Alinear/linear_model_1/linear_model/gender/weighted_sum/zeros_like	ZerosLikeNlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse*'
_output_shapes
:џџџџџџџџџ*
T0
Т
6linear/linear_model_1/linear_model/gender/weighted_sumSelect;linear/linear_model_1/linear_model/gender/weighted_sum/TileAlinear/linear_model_1/linear_model/gender/weighted_sum/zeros_likeNlinear/linear_model_1/linear_model/gender/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
Д
=linear/linear_model_1/linear_model/gender/weighted_sum/Cast_1Cast9linear/linear_model_1/linear_model/gender/SparseReshape:1*
_output_shapes
:*

DstT0*

SrcT0	

Dlinear/linear_model_1/linear_model/gender/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:

Clinear/linear_model_1/linear_model/gender/weighted_sum/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
У
>linear/linear_model_1/linear_model/gender/weighted_sum/Slice_1Slice=linear/linear_model_1/linear_model/gender/weighted_sum/Cast_1Dlinear/linear_model_1/linear_model/gender/weighted_sum/Slice_1/beginClinear/linear_model_1/linear_model/gender/weighted_sum/Slice_1/size*
_output_shapes
:*
Index0*
T0
Є
>linear/linear_model_1/linear_model/gender/weighted_sum/Shape_1Shape6linear/linear_model_1/linear_model/gender/weighted_sum*
T0*
_output_shapes
:

Dlinear/linear_model_1/linear_model/gender/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:

Clinear/linear_model_1/linear_model/gender/weighted_sum/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
Ф
>linear/linear_model_1/linear_model/gender/weighted_sum/Slice_2Slice>linear/linear_model_1/linear_model/gender/weighted_sum/Shape_1Dlinear/linear_model_1/linear_model/gender/weighted_sum/Slice_2/beginClinear/linear_model_1/linear_model/gender/weighted_sum/Slice_2/size*
_output_shapes
:*
Index0*
T0

Blinear/linear_model_1/linear_model/gender/weighted_sum/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Л
=linear/linear_model_1/linear_model/gender/weighted_sum/concatConcatV2>linear/linear_model_1/linear_model/gender/weighted_sum/Slice_1>linear/linear_model_1/linear_model/gender/weighted_sum/Slice_2Blinear/linear_model_1/linear_model/gender/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:
є
@linear/linear_model_1/linear_model/gender/weighted_sum/Reshape_2Reshape6linear/linear_model_1/linear_model/gender/weighted_sum=linear/linear_model_1/linear_model/gender/weighted_sum/concat*'
_output_shapes
:џџџџџџџџџ*
T0
ш
Rlinear/linear_model/gender_X_type/weights/part_0/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@linear/linear_model/gender_X_type/weights/part_0*
valueB"'     
в
Hlinear/linear_model/gender_X_type/weights/part_0/Initializer/zeros/ConstConst*C
_class9
75loc:@linear/linear_model/gender_X_type/weights/part_0*
valueB
 *    *
dtype0*
_output_shapes
: 
з
Blinear/linear_model/gender_X_type/weights/part_0/Initializer/zerosFillRlinear/linear_model/gender_X_type/weights/part_0/Initializer/zeros/shape_as_tensorHlinear/linear_model/gender_X_type/weights/part_0/Initializer/zeros/Const*
T0*C
_class9
75loc:@linear/linear_model/gender_X_type/weights/part_0*
_output_shapes
:	N

0linear/linear_model/gender_X_type/weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape:	N*A
shared_name20linear/linear_model/gender_X_type/weights/part_0*C
_class9
75loc:@linear/linear_model/gender_X_type/weights/part_0
Б
Qlinear/linear_model/gender_X_type/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp0linear/linear_model/gender_X_type/weights/part_0*
_output_shapes
: 

7linear/linear_model/gender_X_type/weights/part_0/AssignAssignVariableOp0linear/linear_model/gender_X_type/weights/part_0Blinear/linear_model/gender_X_type/weights/part_0/Initializer/zeros*C
_class9
75loc:@linear/linear_model/gender_X_type/weights/part_0*
dtype0
ћ
Dlinear/linear_model/gender_X_type/weights/part_0/Read/ReadVariableOpReadVariableOp0linear/linear_model/gender_X_type/weights/part_0*C
_class9
75loc:@linear/linear_model/gender_X_type/weights/part_0*
dtype0*
_output_shapes
:	N

?linear/linear_model_1/linear_model/gender_X_type/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Ы
;linear/linear_model_1/linear_model/gender_X_type/ExpandDims
ExpandDimsPlaceholder_1?linear/linear_model_1/linear_model/gender_X_type/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ

<linear/linear_model_1/linear_model/gender_X_type/SparseCrossSparseCross4linear/linear_model_1/linear_model/gender/ExpandDims;linear/linear_model_1/linear_model/gender_X_type/ExpandDims*
num_bucketsN*
hashed_output(*
out_type0	*
N *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:*
dense_types
2*
hash_keyўпђзь*
sparse_types
 *
internal_type0
З
;linear/linear_model_1/linear_model/gender_X_type/Shape/CastCast>linear/linear_model_1/linear_model/gender_X_type/SparseCross:2*
_output_shapes
:*

DstT0*

SrcT0	

Dlinear/linear_model_1/linear_model/gender_X_type/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Flinear/linear_model_1/linear_model/gender_X_type/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Flinear/linear_model_1/linear_model/gender_X_type/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ї
>linear/linear_model_1/linear_model/gender_X_type/strided_sliceStridedSlice;linear/linear_model_1/linear_model/gender_X_type/Shape/CastDlinear/linear_model_1/linear_model/gender_X_type/strided_slice/stackFlinear/linear_model_1/linear_model/gender_X_type/strided_slice/stack_1Flinear/linear_model_1/linear_model/gender_X_type/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

9linear/linear_model_1/linear_model/gender_X_type/Cast/x/1Const*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
ш
7linear/linear_model_1/linear_model/gender_X_type/Cast/xPack>linear/linear_model_1/linear_model/gender_X_type/strided_slice9linear/linear_model_1/linear_model/gender_X_type/Cast/x/1*
T0*
N*
_output_shapes
:
Њ
5linear/linear_model_1/linear_model/gender_X_type/CastCast7linear/linear_model_1/linear_model/gender_X_type/Cast/x*

SrcT0*
_output_shapes
:*

DstT0	
Г
>linear/linear_model_1/linear_model/gender_X_type/SparseReshapeSparseReshape<linear/linear_model_1/linear_model/gender_X_type/SparseCross>linear/linear_model_1/linear_model/gender_X_type/SparseCross:25linear/linear_model_1/linear_model/gender_X_type/Cast*-
_output_shapes
:џџџџџџџџџ:
С
Glinear/linear_model_1/linear_model/gender_X_type/SparseReshape/IdentityIdentity>linear/linear_model_1/linear_model/gender_X_type/SparseCross:1*
T0	*#
_output_shapes
:џџџџџџџџџ
Б
?linear/linear_model_1/linear_model/gender_X_type/ReadVariableOpReadVariableOp0linear/linear_model/gender_X_type/weights/part_0*
dtype0*
_output_shapes
:	N

Ilinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 

Hlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
е
Clinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SliceSlice@linear/linear_model_1/linear_model/gender_X_type/SparseReshape:1Ilinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice/beginHlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:

Clinear/linear_model_1/linear_model/gender_X_type/weighted_sum/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
ѕ
Blinear/linear_model_1/linear_model/gender_X_type/weighted_sum/ProdProdClinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SliceClinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Const*
_output_shapes
: *
T0	

Nlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 

Klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ѕ
Flinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2GatherV2@linear/linear_model_1/linear_model/gender_X_type/SparseReshape:1Nlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2/indicesKlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0

Dlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Cast/xPackBlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/ProdFlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2*
N*
_output_shapes
:*
T0	
г
Klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseReshapeSparseReshape>linear/linear_model_1/linear_model/gender_X_type/SparseReshape@linear/linear_model_1/linear_model/gender_X_type/SparseReshape:1Dlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Cast/x*-
_output_shapes
:џџџџџџџџџ:
з
Tlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseReshape/IdentityIdentityGlinear/linear_model_1/linear_model/gender_X_type/SparseReshape/Identity*
T0	*#
_output_shapes
:џџџџџџџџџ

Llinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
Ќ
Jlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GreaterEqualGreaterEqualTlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseReshape/IdentityLlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:џџџџџџџџџ
С
Clinear/linear_model_1/linear_model/gender_X_type/weighted_sum/WhereWhereJlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GreaterEqual*'
_output_shapes
:џџџџџџџџџ

Klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:

Elinear/linear_model_1/linear_model/gender_X_type/weighted_sum/ReshapeReshapeClinear/linear_model_1/linear_model/gender_X_type/weighted_sum/WhereKlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:џџџџџџџџџ

Mlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

Hlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2_1GatherV2Klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseReshapeElinear/linear_model_1/linear_model/gender_X_type/weighted_sum/ReshapeMlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2_1/axis*
Tparams0	*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	

Mlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 

Hlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2_2GatherV2Tlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseReshape/IdentityElinear/linear_model_1/linear_model/gender_X_type/weighted_sum/ReshapeMlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2_2/axis*
Tparams0	*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	
Ц
Flinear/linear_model_1/linear_model/gender_X_type/weighted_sum/IdentityIdentityMlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:

Wlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 

elinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsHlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2_1Hlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/GatherV2_2Flinear/linear_model_1/linear_model/gender_X_type/weighted_sum/IdentityWlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
К
ilinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
М
klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
М
klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      

clinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceelinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsilinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/strided_slice/stackklinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ*
T0	*
Index0

Zlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/CastCastclinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0

\linear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/UniqueUniqueglinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/embedding_lookup/axisConst*R
_classH
FDloc:@linear/linear_model_1/linear_model/gender_X_type/ReadVariableOp*
value	B : *
dtype0*
_output_shapes
: 
Ї
flinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/embedding_lookupGatherV2?linear/linear_model_1/linear_model/gender_X_type/ReadVariableOp\linear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/Uniqueklinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/embedding_lookup/axis*'
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0*R
_classH
FDloc:@linear/linear_model_1/linear_model/gender_X_type/ReadVariableOp
П
Ulinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparseSparseSegmentSumflinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/embedding_lookup^linear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/Unique:1Zlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

Mlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ   
М
Glinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Reshape_1Reshapeglinear/linear_model_1/linear_model/gender_X_type/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Mlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ
Ш
Clinear/linear_model_1/linear_model/gender_X_type/weighted_sum/ShapeShapeUlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:

Qlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:

Slinear/linear_model_1/linear_model/gender_X_type/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Slinear/linear_model_1/linear_model/gender_X_type/weighted_sum/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
у
Klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/strided_sliceStridedSliceClinear/linear_model_1/linear_model/gender_X_type/weighted_sum/ShapeQlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/strided_slice/stackSlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/strided_slice/stack_1Slinear/linear_model_1/linear_model/gender_X_type/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Elinear/linear_model_1/linear_model/gender_X_type/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 

Clinear/linear_model_1/linear_model/gender_X_type/weighted_sum/stackPackElinear/linear_model_1/linear_model/gender_X_type/weighted_sum/stack/0Klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:

Blinear/linear_model_1/linear_model/gender_X_type/weighted_sum/TileTileGlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Reshape_1Clinear/linear_model_1/linear_model/gender_X_type/weighted_sum/stack*
T0
*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
о
Hlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/zeros_like	ZerosLikeUlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse*'
_output_shapes
:џџџџџџџџџ*
T0
о
=linear/linear_model_1/linear_model/gender_X_type/weighted_sumSelectBlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/TileHlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/zeros_likeUlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
Т
Dlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Cast_1Cast@linear/linear_model_1/linear_model/gender_X_type/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0

Klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:

Jlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
п
Elinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_1SliceDlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Cast_1Klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_1/beginJlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_1/size*
_output_shapes
:*
Index0*
T0
В
Elinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Shape_1Shape=linear/linear_model_1/linear_model/gender_X_type/weighted_sum*
T0*
_output_shapes
:

Klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:

Jlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
р
Elinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_2SliceElinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Shape_1Klinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_2/beginJlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_2/size*
_output_shapes
:*
Index0*
T0

Ilinear/linear_model_1/linear_model/gender_X_type/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
з
Dlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/concatConcatV2Elinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_1Elinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Slice_2Ilinear/linear_model_1/linear_model/gender_X_type/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:

Glinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Reshape_2Reshape=linear/linear_model_1/linear_model/gender_X_type/weighted_sumDlinear/linear_model_1/linear_model/gender_X_type/weighted_sum/concat*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
:linear/linear_model/hJump/weights/part_0/Initializer/zerosConst*;
_class1
/-loc:@linear/linear_model/hJump/weights/part_0*
valueB*    *
dtype0*
_output_shapes

:
щ
(linear/linear_model/hJump/weights/part_0VarHandleOp*
shape
:*9
shared_name*(linear/linear_model/hJump/weights/part_0*;
_class1
/-loc:@linear/linear_model/hJump/weights/part_0*
dtype0*
_output_shapes
: 
Ё
Ilinear/linear_model/hJump/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp(linear/linear_model/hJump/weights/part_0*
_output_shapes
: 
ѓ
/linear/linear_model/hJump/weights/part_0/AssignAssignVariableOp(linear/linear_model/hJump/weights/part_0:linear/linear_model/hJump/weights/part_0/Initializer/zeros*;
_class1
/-loc:@linear/linear_model/hJump/weights/part_0*
dtype0
т
<linear/linear_model/hJump/weights/part_0/Read/ReadVariableOpReadVariableOp(linear/linear_model/hJump/weights/part_0*
dtype0*
_output_shapes

:*;
_class1
/-loc:@linear/linear_model/hJump/weights/part_0

7linear/linear_model_1/linear_model/hJump/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Л
3linear/linear_model_1/linear_model/hJump/ExpandDims
ExpandDimsPlaceholder_67linear/linear_model_1/linear_model/hJump/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*
T0

.linear/linear_model_1/linear_model/hJump/ShapeShape3linear/linear_model_1/linear_model/hJump/ExpandDims*
T0*
_output_shapes
:

<linear/linear_model_1/linear_model/hJump/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

>linear/linear_model_1/linear_model/hJump/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

>linear/linear_model_1/linear_model/hJump/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
њ
6linear/linear_model_1/linear_model/hJump/strided_sliceStridedSlice.linear/linear_model_1/linear_model/hJump/Shape<linear/linear_model_1/linear_model/hJump/strided_slice/stack>linear/linear_model_1/linear_model/hJump/strided_slice/stack_1>linear/linear_model_1/linear_model/hJump/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
z
8linear/linear_model_1/linear_model/hJump/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
о
6linear/linear_model_1/linear_model/hJump/Reshape/shapePack6linear/linear_model_1/linear_model/hJump/strided_slice8linear/linear_model_1/linear_model/hJump/Reshape/shape/1*
N*
_output_shapes
:*
T0
к
0linear/linear_model_1/linear_model/hJump/ReshapeReshape3linear/linear_model_1/linear_model/hJump/ExpandDims6linear/linear_model_1/linear_model/hJump/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

0linear/linear_model/hJump/weights/ReadVariableOpReadVariableOp(linear/linear_model/hJump/weights/part_0*
dtype0*
_output_shapes

:

!linear/linear_model/hJump/weightsIdentity0linear/linear_model/hJump/weights/ReadVariableOp*
T0*
_output_shapes

:
Ц
5linear/linear_model_1/linear_model/hJump/weighted_sumMatMul0linear/linear_model_1/linear_model/hJump/Reshape!linear/linear_model/hJump/weights*'
_output_shapes
:џџџџџџџџџ*
T0
Ю
;linear/linear_model/height/weights/part_0/Initializer/zerosConst*<
_class2
0.loc:@linear/linear_model/height/weights/part_0*
valueB*    *
dtype0*
_output_shapes

:
ь
)linear/linear_model/height/weights/part_0VarHandleOp*
shape
:*:
shared_name+)linear/linear_model/height/weights/part_0*<
_class2
0.loc:@linear/linear_model/height/weights/part_0*
dtype0*
_output_shapes
: 
Ѓ
Jlinear/linear_model/height/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp)linear/linear_model/height/weights/part_0*
_output_shapes
: 
ї
0linear/linear_model/height/weights/part_0/AssignAssignVariableOp)linear/linear_model/height/weights/part_0;linear/linear_model/height/weights/part_0/Initializer/zeros*<
_class2
0.loc:@linear/linear_model/height/weights/part_0*
dtype0
х
=linear/linear_model/height/weights/part_0/Read/ReadVariableOpReadVariableOp)linear/linear_model/height/weights/part_0*<
_class2
0.loc:@linear/linear_model/height/weights/part_0*
dtype0*
_output_shapes

:

8linear/linear_model_1/linear_model/height/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Н
4linear/linear_model_1/linear_model/height/ExpandDims
ExpandDimsPlaceholder_48linear/linear_model_1/linear_model/height/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ

/linear/linear_model_1/linear_model/height/ShapeShape4linear/linear_model_1/linear_model/height/ExpandDims*
T0*
_output_shapes
:

=linear/linear_model_1/linear_model/height/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

?linear/linear_model_1/linear_model/height/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:

?linear/linear_model_1/linear_model/height/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
џ
7linear/linear_model_1/linear_model/height/strided_sliceStridedSlice/linear/linear_model_1/linear_model/height/Shape=linear/linear_model_1/linear_model/height/strided_slice/stack?linear/linear_model_1/linear_model/height/strided_slice/stack_1?linear/linear_model_1/linear_model/height/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
{
9linear/linear_model_1/linear_model/height/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
с
7linear/linear_model_1/linear_model/height/Reshape/shapePack7linear/linear_model_1/linear_model/height/strided_slice9linear/linear_model_1/linear_model/height/Reshape/shape/1*
T0*
N*
_output_shapes
:
н
1linear/linear_model_1/linear_model/height/ReshapeReshape4linear/linear_model_1/linear_model/height/ExpandDims7linear/linear_model_1/linear_model/height/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

1linear/linear_model/height/weights/ReadVariableOpReadVariableOp)linear/linear_model/height/weights/part_0*
dtype0*
_output_shapes

:

"linear/linear_model/height/weightsIdentity1linear/linear_model/height/weights/ReadVariableOp*
_output_shapes

:*
T0
Щ
6linear/linear_model_1/linear_model/height/weighted_sumMatMul1linear/linear_model_1/linear_model/height/Reshape"linear/linear_model/height/weights*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
:linear/linear_model/tJump/weights/part_0/Initializer/zerosConst*
dtype0*
_output_shapes

:*;
_class1
/-loc:@linear/linear_model/tJump/weights/part_0*
valueB*    
щ
(linear/linear_model/tJump/weights/part_0VarHandleOp*
shape
:*9
shared_name*(linear/linear_model/tJump/weights/part_0*;
_class1
/-loc:@linear/linear_model/tJump/weights/part_0*
dtype0*
_output_shapes
: 
Ё
Ilinear/linear_model/tJump/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp(linear/linear_model/tJump/weights/part_0*
_output_shapes
: 
ѓ
/linear/linear_model/tJump/weights/part_0/AssignAssignVariableOp(linear/linear_model/tJump/weights/part_0:linear/linear_model/tJump/weights/part_0/Initializer/zeros*;
_class1
/-loc:@linear/linear_model/tJump/weights/part_0*
dtype0
т
<linear/linear_model/tJump/weights/part_0/Read/ReadVariableOpReadVariableOp(linear/linear_model/tJump/weights/part_0*;
_class1
/-loc:@linear/linear_model/tJump/weights/part_0*
dtype0*
_output_shapes

:

7linear/linear_model_1/linear_model/tJump/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Л
3linear/linear_model_1/linear_model/tJump/ExpandDims
ExpandDimsPlaceholder_57linear/linear_model_1/linear_model/tJump/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ

.linear/linear_model_1/linear_model/tJump/ShapeShape3linear/linear_model_1/linear_model/tJump/ExpandDims*
T0*
_output_shapes
:

<linear/linear_model_1/linear_model/tJump/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

>linear/linear_model_1/linear_model/tJump/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

>linear/linear_model_1/linear_model/tJump/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
њ
6linear/linear_model_1/linear_model/tJump/strided_sliceStridedSlice.linear/linear_model_1/linear_model/tJump/Shape<linear/linear_model_1/linear_model/tJump/strided_slice/stack>linear/linear_model_1/linear_model/tJump/strided_slice/stack_1>linear/linear_model_1/linear_model/tJump/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
z
8linear/linear_model_1/linear_model/tJump/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
о
6linear/linear_model_1/linear_model/tJump/Reshape/shapePack6linear/linear_model_1/linear_model/tJump/strided_slice8linear/linear_model_1/linear_model/tJump/Reshape/shape/1*
T0*
N*
_output_shapes
:
к
0linear/linear_model_1/linear_model/tJump/ReshapeReshape3linear/linear_model_1/linear_model/tJump/ExpandDims6linear/linear_model_1/linear_model/tJump/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

0linear/linear_model/tJump/weights/ReadVariableOpReadVariableOp(linear/linear_model/tJump/weights/part_0*
dtype0*
_output_shapes

:

!linear/linear_model/tJump/weightsIdentity0linear/linear_model/tJump/weights/ReadVariableOp*
T0*
_output_shapes

:
Ц
5linear/linear_model_1/linear_model/tJump/weighted_sumMatMul0linear/linear_model_1/linear_model/tJump/Reshape!linear/linear_model/tJump/weights*
T0*'
_output_shapes
:џџџџџџџџџ
Ъ
9linear/linear_model/type/weights/part_0/Initializer/zerosConst*:
_class0
.,loc:@linear/linear_model/type/weights/part_0*
valueB*    *
dtype0*
_output_shapes

:
ц
'linear/linear_model/type/weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape
:*8
shared_name)'linear/linear_model/type/weights/part_0*:
_class0
.,loc:@linear/linear_model/type/weights/part_0

Hlinear/linear_model/type/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/type/weights/part_0*
_output_shapes
: 
я
.linear/linear_model/type/weights/part_0/AssignAssignVariableOp'linear/linear_model/type/weights/part_09linear/linear_model/type/weights/part_0/Initializer/zeros*:
_class0
.,loc:@linear/linear_model/type/weights/part_0*
dtype0
п
;linear/linear_model/type/weights/part_0/Read/ReadVariableOpReadVariableOp'linear/linear_model/type/weights/part_0*:
_class0
.,loc:@linear/linear_model/type/weights/part_0*
dtype0*
_output_shapes

:

Flinear/linear_model_1/linear_model/type/to_sparse_input/ignore_value/xConst*
dtype0*
_output_shapes
: *
valueB B 

@linear/linear_model_1/linear_model/type/to_sparse_input/NotEqualNotEqual;linear/linear_model_1/linear_model/gender_X_type/ExpandDimsFlinear/linear_model_1/linear_model/type/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:џџџџџџџџџ
Г
?linear/linear_model_1/linear_model/type/to_sparse_input/indicesWhere@linear/linear_model_1/linear_model/type/to_sparse_input/NotEqual*'
_output_shapes
:џџџџџџџџџ

>linear/linear_model_1/linear_model/type/to_sparse_input/valuesGatherNd;linear/linear_model_1/linear_model/gender_X_type/ExpandDims?linear/linear_model_1/linear_model/type/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ
О
Clinear/linear_model_1/linear_model/type/to_sparse_input/dense_shapeShape;linear/linear_model_1/linear_model/gender_X_type/ExpandDims*
T0*
out_type0	*
_output_shapes
:

9linear/linear_model_1/linear_model/type/type_lookup/ConstConst*І
valueBB sarituri ca mingeaB sarituri ca mingea cu greutateB sarituri duble greutateB sarituri dubleB sarituri simpleB sarituri simple cu greutate*
dtype0*
_output_shapes
:
z
8linear/linear_model_1/linear_model/type/type_lookup/SizeConst*
value	B :*
dtype0*
_output_shapes
: 

?linear/linear_model_1/linear_model/type/type_lookup/range/startConst*
dtype0*
_output_shapes
: *
value	B : 

?linear/linear_model_1/linear_model/type/type_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

9linear/linear_model_1/linear_model/type/type_lookup/rangeRange?linear/linear_model_1/linear_model/type/type_lookup/range/start8linear/linear_model_1/linear_model/type/type_lookup/Size?linear/linear_model_1/linear_model/type/type_lookup/range/delta*
_output_shapes
:
В
;linear/linear_model_1/linear_model/type/type_lookup/ToInt64Cast9linear/linear_model_1/linear_model/type/type_lookup/range*

SrcT0*
_output_shapes
:*

DstT0	

>linear/linear_model_1/linear_model/type/type_lookup/hash_tableHashTableV2*
	key_dtype0*
value_dtype0	*
_output_shapes
: 

Dlinear/linear_model_1/linear_model/type/type_lookup/hash_table/ConstConst*
dtype0	*
_output_shapes
: *
valueB	 R
џџџџџџџџџ
Ў
Ilinear/linear_model_1/linear_model/type/type_lookup/hash_table/table_initInitializeTableV2>linear/linear_model_1/linear_model/type/type_lookup/hash_table9linear/linear_model_1/linear_model/type/type_lookup/Const;linear/linear_model_1/linear_model/type/type_lookup/ToInt64*

Tkey0*

Tval0	
а
9linear/linear_model_1/linear_model/type/hash_table_LookupLookupTableFindV2>linear/linear_model_1/linear_model/type/type_lookup/hash_table>linear/linear_model_1/linear_model/type/to_sparse_input/valuesDlinear/linear_model_1/linear_model/type/type_lookup/hash_table/Const*#
_output_shapes
:џџџџџџџџџ*	
Tin0*

Tout0	
Г
2linear/linear_model_1/linear_model/type/Shape/CastCastClinear/linear_model_1/linear_model/type/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	

;linear/linear_model_1/linear_model/type/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

=linear/linear_model_1/linear_model/type/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

=linear/linear_model_1/linear_model/type/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
њ
5linear/linear_model_1/linear_model/type/strided_sliceStridedSlice2linear/linear_model_1/linear_model/type/Shape/Cast;linear/linear_model_1/linear_model/type/strided_slice/stack=linear/linear_model_1/linear_model/type/strided_slice/stack_1=linear/linear_model_1/linear_model/type/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
{
0linear/linear_model_1/linear_model/type/Cast/x/1Const*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Э
.linear/linear_model_1/linear_model/type/Cast/xPack5linear/linear_model_1/linear_model/type/strided_slice0linear/linear_model_1/linear_model/type/Cast/x/1*
T0*
N*
_output_shapes
:

,linear/linear_model_1/linear_model/type/CastCast.linear/linear_model_1/linear_model/type/Cast/x*

SrcT0*
_output_shapes
:*

DstT0	
Љ
5linear/linear_model_1/linear_model/type/SparseReshapeSparseReshape?linear/linear_model_1/linear_model/type/to_sparse_input/indicesClinear/linear_model_1/linear_model/type/to_sparse_input/dense_shape,linear/linear_model_1/linear_model/type/Cast*-
_output_shapes
:џџџџџџџџџ:
Г
>linear/linear_model_1/linear_model/type/SparseReshape/IdentityIdentity9linear/linear_model_1/linear_model/type/hash_table_Lookup*#
_output_shapes
:џџџџџџџџџ*
T0	

6linear/linear_model_1/linear_model/type/ReadVariableOpReadVariableOp'linear/linear_model/type/weights/part_0*
dtype0*
_output_shapes

:

@linear/linear_model_1/linear_model/type/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:

?linear/linear_model_1/linear_model/type/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Б
:linear/linear_model_1/linear_model/type/weighted_sum/SliceSlice7linear/linear_model_1/linear_model/type/SparseReshape:1@linear/linear_model_1/linear_model/type/weighted_sum/Slice/begin?linear/linear_model_1/linear_model/type/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:

:linear/linear_model_1/linear_model/type/weighted_sum/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
к
9linear/linear_model_1/linear_model/type/weighted_sum/ProdProd:linear/linear_model_1/linear_model/type/weighted_sum/Slice:linear/linear_model_1/linear_model/type/weighted_sum/Const*
T0	*
_output_shapes
: 

Elinear/linear_model_1/linear_model/type/weighted_sum/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :

Blinear/linear_model_1/linear_model/type/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
б
=linear/linear_model_1/linear_model/type/weighted_sum/GatherV2GatherV27linear/linear_model_1/linear_model/type/SparseReshape:1Elinear/linear_model_1/linear_model/type/weighted_sum/GatherV2/indicesBlinear/linear_model_1/linear_model/type/weighted_sum/GatherV2/axis*
Tindices0*
Tparams0	*
_output_shapes
: *
Taxis0
ы
;linear/linear_model_1/linear_model/type/weighted_sum/Cast/xPack9linear/linear_model_1/linear_model/type/weighted_sum/Prod=linear/linear_model_1/linear_model/type/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
Џ
Blinear/linear_model_1/linear_model/type/weighted_sum/SparseReshapeSparseReshape5linear/linear_model_1/linear_model/type/SparseReshape7linear/linear_model_1/linear_model/type/SparseReshape:1;linear/linear_model_1/linear_model/type/weighted_sum/Cast/x*-
_output_shapes
:џџџџџџџџџ:
Х
Klinear/linear_model_1/linear_model/type/weighted_sum/SparseReshape/IdentityIdentity>linear/linear_model_1/linear_model/type/SparseReshape/Identity*
T0	*#
_output_shapes
:џџџџџџџџџ

Clinear/linear_model_1/linear_model/type/weighted_sum/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 

Alinear/linear_model_1/linear_model/type/weighted_sum/GreaterEqualGreaterEqualKlinear/linear_model_1/linear_model/type/weighted_sum/SparseReshape/IdentityClinear/linear_model_1/linear_model/type/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:џџџџџџџџџ
Џ
:linear/linear_model_1/linear_model/type/weighted_sum/WhereWhereAlinear/linear_model_1/linear_model/type/weighted_sum/GreaterEqual*'
_output_shapes
:џџџџџџџџџ

Blinear/linear_model_1/linear_model/type/weighted_sum/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ѕ
<linear/linear_model_1/linear_model/type/weighted_sum/ReshapeReshape:linear/linear_model_1/linear_model/type/weighted_sum/WhereBlinear/linear_model_1/linear_model/type/weighted_sum/Reshape/shape*#
_output_shapes
:џџџџџџџџџ*
T0	

Dlinear/linear_model_1/linear_model/type/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ш
?linear/linear_model_1/linear_model/type/weighted_sum/GatherV2_1GatherV2Blinear/linear_model_1/linear_model/type/weighted_sum/SparseReshape<linear/linear_model_1/linear_model/type/weighted_sum/ReshapeDlinear/linear_model_1/linear_model/type/weighted_sum/GatherV2_1/axis*
Tindices0	*
Tparams0	*'
_output_shapes
:џџџџџџџџџ*
Taxis0

Dlinear/linear_model_1/linear_model/type/weighted_sum/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
э
?linear/linear_model_1/linear_model/type/weighted_sum/GatherV2_2GatherV2Klinear/linear_model_1/linear_model/type/weighted_sum/SparseReshape/Identity<linear/linear_model_1/linear_model/type/weighted_sum/ReshapeDlinear/linear_model_1/linear_model/type/weighted_sum/GatherV2_2/axis*
Tparams0	*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	
Д
=linear/linear_model_1/linear_model/type/weighted_sum/IdentityIdentityDlinear/linear_model_1/linear_model/type/weighted_sum/SparseReshape:1*
_output_shapes
:*
T0	

Nlinear/linear_model_1/linear_model/type/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
у
\linear/linear_model_1/linear_model/type/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?linear/linear_model_1/linear_model/type/weighted_sum/GatherV2_1?linear/linear_model_1/linear_model/type/weighted_sum/GatherV2_2=linear/linear_model_1/linear_model/type/weighted_sum/IdentityNlinear/linear_model_1/linear_model/type/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
Б
`linear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
Г
blinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
Г
blinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ч
Zlinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlice\linear/linear_model_1/linear_model/type/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows`linear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/strided_slice/stackblinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1blinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
Index0*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:џџџџџџџџџ
ђ
Qlinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/CastCastZlinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
њ
Slinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/UniqueUnique^linear/linear_model_1/linear_model/type/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
я
blinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/embedding_lookup/axisConst*I
_class?
=;loc:@linear/linear_model_1/linear_model/type/ReadVariableOp*
value	B : *
dtype0*
_output_shapes
: 
њ
]linear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/embedding_lookupGatherV26linear/linear_model_1/linear_model/type/ReadVariableOpSlinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/Uniqueblinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/embedding_lookup/axis*
Tindices0	*
Tparams0*I
_class?
=;loc:@linear/linear_model_1/linear_model/type/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ*
Taxis0

Llinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparseSparseSegmentSum]linear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/embedding_lookupUlinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/Unique:1Qlinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:џџџџџџџџџ

Dlinear/linear_model_1/linear_model/type/weighted_sum/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ   
Ё
>linear/linear_model_1/linear_model/type/weighted_sum/Reshape_1Reshape^linear/linear_model_1/linear_model/type/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Dlinear/linear_model_1/linear_model/type/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:џџџџџџџџџ
Ж
:linear/linear_model_1/linear_model/type/weighted_sum/ShapeShapeLlinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse*
_output_shapes
:*
T0

Hlinear/linear_model_1/linear_model/type/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:

Jlinear/linear_model_1/linear_model/type/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Jlinear/linear_model_1/linear_model/type/weighted_sum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ж
Blinear/linear_model_1/linear_model/type/weighted_sum/strided_sliceStridedSlice:linear/linear_model_1/linear_model/type/weighted_sum/ShapeHlinear/linear_model_1/linear_model/type/weighted_sum/strided_slice/stackJlinear/linear_model_1/linear_model/type/weighted_sum/strided_slice/stack_1Jlinear/linear_model_1/linear_model/type/weighted_sum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
~
<linear/linear_model_1/linear_model/type/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
ђ
:linear/linear_model_1/linear_model/type/weighted_sum/stackPack<linear/linear_model_1/linear_model/type/weighted_sum/stack/0Blinear/linear_model_1/linear_model/type/weighted_sum/strided_slice*
N*
_output_shapes
:*
T0
ј
9linear/linear_model_1/linear_model/type/weighted_sum/TileTile>linear/linear_model_1/linear_model/type/weighted_sum/Reshape_1:linear/linear_model_1/linear_model/type/weighted_sum/stack*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

Ь
?linear/linear_model_1/linear_model/type/weighted_sum/zeros_like	ZerosLikeLlinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
К
4linear/linear_model_1/linear_model/type/weighted_sumSelect9linear/linear_model_1/linear_model/type/weighted_sum/Tile?linear/linear_model_1/linear_model/type/weighted_sum/zeros_likeLlinear/linear_model_1/linear_model/type/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:џџџџџџџџџ
А
;linear/linear_model_1/linear_model/type/weighted_sum/Cast_1Cast7linear/linear_model_1/linear_model/type/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0

Blinear/linear_model_1/linear_model/type/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:

Alinear/linear_model_1/linear_model/type/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Л
<linear/linear_model_1/linear_model/type/weighted_sum/Slice_1Slice;linear/linear_model_1/linear_model/type/weighted_sum/Cast_1Blinear/linear_model_1/linear_model/type/weighted_sum/Slice_1/beginAlinear/linear_model_1/linear_model/type/weighted_sum/Slice_1/size*
_output_shapes
:*
Index0*
T0
 
<linear/linear_model_1/linear_model/type/weighted_sum/Shape_1Shape4linear/linear_model_1/linear_model/type/weighted_sum*
T0*
_output_shapes
:

Blinear/linear_model_1/linear_model/type/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:

Alinear/linear_model_1/linear_model/type/weighted_sum/Slice_2/sizeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
М
<linear/linear_model_1/linear_model/type/weighted_sum/Slice_2Slice<linear/linear_model_1/linear_model/type/weighted_sum/Shape_1Blinear/linear_model_1/linear_model/type/weighted_sum/Slice_2/beginAlinear/linear_model_1/linear_model/type/weighted_sum/Slice_2/size*
_output_shapes
:*
Index0*
T0

@linear/linear_model_1/linear_model/type/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Г
;linear/linear_model_1/linear_model/type/weighted_sum/concatConcatV2<linear/linear_model_1/linear_model/type/weighted_sum/Slice_1<linear/linear_model_1/linear_model/type/weighted_sum/Slice_2@linear/linear_model_1/linear_model/type/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:
ю
>linear/linear_model_1/linear_model/type/weighted_sum/Reshape_2Reshape4linear/linear_model_1/linear_model/type/weighted_sum;linear/linear_model_1/linear_model/type/weighted_sum/concat*
T0*'
_output_shapes
:џџџџџџџџџ
Ю
;linear/linear_model/weight/weights/part_0/Initializer/zerosConst*<
_class2
0.loc:@linear/linear_model/weight/weights/part_0*
valueB*    *
dtype0*
_output_shapes

:
ь
)linear/linear_model/weight/weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape
:*:
shared_name+)linear/linear_model/weight/weights/part_0*<
_class2
0.loc:@linear/linear_model/weight/weights/part_0
Ѓ
Jlinear/linear_model/weight/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp)linear/linear_model/weight/weights/part_0*
_output_shapes
: 
ї
0linear/linear_model/weight/weights/part_0/AssignAssignVariableOp)linear/linear_model/weight/weights/part_0;linear/linear_model/weight/weights/part_0/Initializer/zeros*<
_class2
0.loc:@linear/linear_model/weight/weights/part_0*
dtype0
х
=linear/linear_model/weight/weights/part_0/Read/ReadVariableOpReadVariableOp)linear/linear_model/weight/weights/part_0*<
_class2
0.loc:@linear/linear_model/weight/weights/part_0*
dtype0*
_output_shapes

:

8linear/linear_model_1/linear_model/weight/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
Н
4linear/linear_model_1/linear_model/weight/ExpandDims
ExpandDimsPlaceholder_38linear/linear_model_1/linear_model/weight/ExpandDims/dim*
T0*'
_output_shapes
:џџџџџџџџџ

/linear/linear_model_1/linear_model/weight/ShapeShape4linear/linear_model_1/linear_model/weight/ExpandDims*
T0*
_output_shapes
:

=linear/linear_model_1/linear_model/weight/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

?linear/linear_model_1/linear_model/weight/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

?linear/linear_model_1/linear_model/weight/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
џ
7linear/linear_model_1/linear_model/weight/strided_sliceStridedSlice/linear/linear_model_1/linear_model/weight/Shape=linear/linear_model_1/linear_model/weight/strided_slice/stack?linear/linear_model_1/linear_model/weight/strided_slice/stack_1?linear/linear_model_1/linear_model/weight/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
{
9linear/linear_model_1/linear_model/weight/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
с
7linear/linear_model_1/linear_model/weight/Reshape/shapePack7linear/linear_model_1/linear_model/weight/strided_slice9linear/linear_model_1/linear_model/weight/Reshape/shape/1*
T0*
N*
_output_shapes
:
н
1linear/linear_model_1/linear_model/weight/ReshapeReshape4linear/linear_model_1/linear_model/weight/ExpandDims7linear/linear_model_1/linear_model/weight/Reshape/shape*
T0*'
_output_shapes
:џџџџџџџџџ

1linear/linear_model/weight/weights/ReadVariableOpReadVariableOp)linear/linear_model/weight/weights/part_0*
dtype0*
_output_shapes

:

"linear/linear_model/weight/weightsIdentity1linear/linear_model/weight/weights/ReadVariableOp*
T0*
_output_shapes

:
Щ
6linear/linear_model_1/linear_model/weight/weighted_sumMatMul1linear/linear_model_1/linear_model/weight/Reshape"linear/linear_model/weight/weights*
T0*'
_output_shapes
:џџџџџџџџџ
э
7linear/linear_model_1/linear_model/weighted_sum_no_biasAddNHlinear/linear_model_1/linear_model/age_bucketized/weighted_sum/Reshape_2@linear/linear_model_1/linear_model/gender/weighted_sum/Reshape_2Glinear/linear_model_1/linear_model/gender_X_type/weighted_sum/Reshape_25linear/linear_model_1/linear_model/hJump/weighted_sum6linear/linear_model_1/linear_model/height/weighted_sum5linear/linear_model_1/linear_model/tJump/weighted_sum>linear/linear_model_1/linear_model/type/weighted_sum/Reshape_26linear/linear_model_1/linear_model/weight/weighted_sum*
T0*
N*'
_output_shapes
:џџџџџџџџџ
Т
9linear/linear_model/bias_weights/part_0/Initializer/zerosConst*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
valueB*    *
dtype0*
_output_shapes
:
т
'linear/linear_model/bias_weights/part_0VarHandleOp*
shape:*8
shared_name)'linear/linear_model/bias_weights/part_0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
: 

Hlinear/linear_model/bias_weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/bias_weights/part_0*
_output_shapes
: 
я
.linear/linear_model/bias_weights/part_0/AssignAssignVariableOp'linear/linear_model/bias_weights/part_09linear/linear_model/bias_weights/part_0/Initializer/zeros*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0
л
;linear/linear_model/bias_weights/part_0/Read/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:

/linear/linear_model/bias_weights/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:

 linear/linear_model/bias_weightsIdentity/linear/linear_model/bias_weights/ReadVariableOp*
T0*
_output_shapes
:
Ч
/linear/linear_model_1/linear_model/weighted_sumBiasAdd7linear/linear_model_1/linear_model/weighted_sum_no_bias linear/linear_model/bias_weights*'
_output_shapes
:џџџџџџџџџ*
T0
y
linear/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
d
linear/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
linear/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
linear/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
й
linear/strided_sliceStridedSlicelinear/ReadVariableOplinear/strided_slice/stacklinear/strided_slice/stack_1linear/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
\
linear/bias/tagsConst*
valueB Blinear/bias*
dtype0*
_output_shapes
: 
e
linear/biasScalarSummarylinear/bias/tagslinear/strided_slice*
T0*
_output_shapes
: 

linear/Reshape/ReadVariableOpReadVariableOp)linear/linear_model/gender/weights/part_0*
dtype0*
_output_shapes

:
g
linear/Reshape/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
s
linear/ReshapeReshapelinear/Reshape/ReadVariableOplinear/Reshape/shape*
T0*
_output_shapes
:

linear/Reshape_1/ReadVariableOpReadVariableOp)linear/linear_model/height/weights/part_0*
dtype0*
_output_shapes

:
i
linear/Reshape_1/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
y
linear/Reshape_1Reshapelinear/Reshape_1/ReadVariableOplinear/Reshape_1/shape*
T0*
_output_shapes
:

linear/Reshape_2/ReadVariableOpReadVariableOp'linear/linear_model/type/weights/part_0*
dtype0*
_output_shapes

:
i
linear/Reshape_2/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
y
linear/Reshape_2Reshapelinear/Reshape_2/ReadVariableOplinear/Reshape_2/shape*
T0*
_output_shapes
:

linear/Reshape_3/ReadVariableOpReadVariableOp(linear/linear_model/tJump/weights/part_0*
dtype0*
_output_shapes

:
i
linear/Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
y
linear/Reshape_3Reshapelinear/Reshape_3/ReadVariableOplinear/Reshape_3/shape*
T0*
_output_shapes
:

linear/Reshape_4/ReadVariableOpReadVariableOp)linear/linear_model/weight/weights/part_0*
dtype0*
_output_shapes

:
i
linear/Reshape_4/shapeConst*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
y
linear/Reshape_4Reshapelinear/Reshape_4/ReadVariableOplinear/Reshape_4/shape*
T0*
_output_shapes
:

linear/Reshape_5/ReadVariableOpReadVariableOp(linear/linear_model/hJump/weights/part_0*
dtype0*
_output_shapes

:
i
linear/Reshape_5/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
y
linear/Reshape_5Reshapelinear/Reshape_5/ReadVariableOplinear/Reshape_5/shape*
T0*
_output_shapes
:

linear/Reshape_6/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0*
dtype0*
_output_shapes

:
i
linear/Reshape_6/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
y
linear/Reshape_6Reshapelinear/Reshape_6/ReadVariableOplinear/Reshape_6/shape*
T0*
_output_shapes
:

linear/Reshape_7/ReadVariableOpReadVariableOp0linear/linear_model/gender_X_type/weights/part_0*
dtype0*
_output_shapes
:	N
i
linear/Reshape_7/shapeConst*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
z
linear/Reshape_7Reshapelinear/Reshape_7/ReadVariableOplinear/Reshape_7/shape*
T0*
_output_shapes	
:N
T
linear/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ъ
linear/concatConcatV2linear/Reshapelinear/Reshape_1linear/Reshape_2linear/Reshape_3linear/Reshape_4linear/Reshape_5linear/Reshape_6linear/Reshape_7linear/concat/axis*
T0*
N*
_output_shapes	
:ЇN
^
linear/zero_fraction/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    
s
linear/zero_fraction/EqualEquallinear/concatlinear/zero_fraction/zero*
T0*
_output_shapes	
:ЇN
r
linear/zero_fraction/CastCastlinear/zero_fraction/Equal*
_output_shapes	
:ЇN*

DstT0*

SrcT0

d
linear/zero_fraction/ConstConst*
valueB: *
dtype0*
_output_shapes
:
y
linear/zero_fraction/MeanMeanlinear/zero_fraction/Castlinear/zero_fraction/Const*
_output_shapes
: *
T0

$linear/fraction_of_zero_weights/tagsConst*0
value'B% Blinear/fraction_of_zero_weights*
dtype0*
_output_shapes
: 

linear/fraction_of_zero_weightsScalarSummary$linear/fraction_of_zero_weights/tagslinear/zero_fraction/Mean*
T0*
_output_shapes
: 
`
linear/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ѕ
linear/zero_fraction_1/EqualEqual/linear/linear_model_1/linear_model/weighted_sumlinear/zero_fraction_1/zero*'
_output_shapes
:џџџџџџџџџ*
T0

linear/zero_fraction_1/CastCastlinear/zero_fraction_1/Equal*

SrcT0
*'
_output_shapes
:џџџџџџџџџ*

DstT0
m
linear/zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

linear/zero_fraction_1/MeanMeanlinear/zero_fraction_1/Castlinear/zero_fraction_1/Const*
T0*
_output_shapes
: 

*linear/linear/fraction_of_zero_values/tagsConst*6
value-B+ B%linear/linear/fraction_of_zero_values*
dtype0*
_output_shapes
: 
 
%linear/linear/fraction_of_zero_valuesScalarSummary*linear/linear/fraction_of_zero_values/tagslinear/zero_fraction_1/Mean*
T0*
_output_shapes
: 
u
linear/linear/activation/tagConst*)
value B Blinear/linear/activation*
dtype0*
_output_shapes
: 

linear/linear/activationHistogramSummarylinear/linear/activation/tag/linear/linear_model_1/linear_model/weighted_sum*
_output_shapes
: 

addAdddnn/logits/BiasAdd/linear/linear_model_1/linear_model/weighted_sum*
T0*'
_output_shapes
:џџџџџџџџџ
P
head/predictions/logits/ShapeShapeadd*
T0*
_output_shapes
:
s
1head/predictions/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
c
[head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
T
Lhead/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
[
head/predictions/logisticSigmoidadd*
T0*'
_output_shapes
:џџџџџџџџџ
_
head/predictions/zeros_like	ZerosLikeadd*
T0*'
_output_shapes
:џџџџџџџџџ
q
&head/predictions/two_class_logits/axisConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
В
!head/predictions/two_class_logitsConcatV2head/predictions/zeros_likeadd&head/predictions/two_class_logits/axis*
T0*
N*'
_output_shapes
:џџџџџџџџџ
~
head/predictions/probabilitiesSoftmax!head/predictions/two_class_logits*'
_output_shapes
:џџџџџџџџџ*
T0
o
$head/predictions/class_ids/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ

head/predictions/class_idsArgMax!head/predictions/two_class_logits$head/predictions/class_ids/dimension*
T0*#
_output_shapes
:џџџџџџџџџ
j
head/predictions/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ

head/predictions/ExpandDims
ExpandDimshead/predictions/class_idshead/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:џџџџџџџџџ
w
head/predictions/str_classesAsStringhead/predictions/ExpandDims*
T0	*'
_output_shapes
:џџџџџџџџџ
X

head/ShapeShapehead/predictions/probabilities*
T0*
_output_shapes
:
b
head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
d
head/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
d
head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ц
head/strided_sliceStridedSlice
head/Shapehead/strided_slice/stackhead/strided_slice/stack_1head/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
R
head/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
R
head/range/limitConst*
value	B :*
dtype0*
_output_shapes
: 
R
head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
e

head/rangeRangehead/range/starthead/range/limithead/range/delta*
_output_shapes
:
J
head/AsStringAsString
head/range*
T0*
_output_shapes
:
U
head/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
j
head/ExpandDims
ExpandDimshead/AsStringhead/ExpandDims/dim*
T0*
_output_shapes

:
W
head/Tile/multiples/1Const*
dtype0*
_output_shapes
: *
value	B :
t
head/Tile/multiplesPackhead/strided_slicehead/Tile/multiples/1*
T0*
N*
_output_shapes
:
i
	head/TileTilehead/ExpandDimshead/Tile/multiples*
T0*'
_output_shapes
:џџџџџџџџџ

initNoOp
ѓ
init_all_tablesNoOp`^dnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/table_init\^dnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/hash_table/table_initN^linear/linear_model_1/linear_model/gender/gender_lookup/hash_table/table_initJ^linear/linear_model_1/linear_model/type/type_lookup/hash_table/table_init

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

save/Read/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0*
dtype0*
_output_shapes

:
\
save/IdentityIdentitysave/Read/ReadVariableOp*
_output_shapes

:*
T0
b
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes

:
~
save/Read_1/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
\
save/Identity_2Identitysave/Read_1/ReadVariableOp*
T0*
_output_shapes
:
`
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes
:

save/Read_2/ReadVariableOpReadVariableOp)linear/linear_model/gender/weights/part_0*
dtype0*
_output_shapes

:
`
save/Identity_4Identitysave/Read_2/ReadVariableOp*
T0*
_output_shapes

:
d
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
_output_shapes

:*
T0

save/Read_3/ReadVariableOpReadVariableOp0linear/linear_model/gender_X_type/weights/part_0*
dtype0*
_output_shapes
:	N
a
save/Identity_6Identitysave/Read_3/ReadVariableOp*
T0*
_output_shapes
:	N
e
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
_output_shapes
:	N*
T0

save/Read_4/ReadVariableOpReadVariableOp(linear/linear_model/hJump/weights/part_0*
dtype0*
_output_shapes

:
`
save/Identity_8Identitysave/Read_4/ReadVariableOp*
T0*
_output_shapes

:
d
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
T0*
_output_shapes

:

save/Read_5/ReadVariableOpReadVariableOp)linear/linear_model/height/weights/part_0*
dtype0*
_output_shapes

:
a
save/Identity_10Identitysave/Read_5/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:

save/Read_6/ReadVariableOpReadVariableOp(linear/linear_model/tJump/weights/part_0*
dtype0*
_output_shapes

:
a
save/Identity_12Identitysave/Read_6/ReadVariableOp*
_output_shapes

:*
T0
f
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
_output_shapes

:*
T0

save/Read_7/ReadVariableOpReadVariableOp'linear/linear_model/type/weights/part_0*
dtype0*
_output_shapes

:
a
save/Identity_14Identitysave/Read_7/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
T0*
_output_shapes

:

save/Read_8/ReadVariableOpReadVariableOp)linear/linear_model/weight/weights/part_0*
dtype0*
_output_shapes

:
a
save/Identity_16Identitysave/Read_8/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
T0*
_output_shapes

:

save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_b06071c8b1c64c1e9c26987a0b698f06/part
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
ъ
save/SaveV2/tensor_namesConst"/device:CPU:0*
valueBBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/hiddenlayer_3/biasBdnn/hiddenlayer_3/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
љ
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB	100 0,100B8 100 0,8:0,100B70 0,70B100 70 0,100:0,70B48 0,48B70 48 0,70:0,48B34 0,34B48 34 0,48:0,34B1 0,1B34 1 0,34:0,1B *
dtype0*
_output_shapes
:
ў
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/read"dnn/hiddenlayer_2/bias/part_0/read$dnn/hiddenlayer_2/kernel/part_0/read"dnn/hiddenlayer_3/bias/part_0/read$dnn/hiddenlayer_3/kernel/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 

save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 

save/Read_9/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
p
save/Identity_18Identitysave/Read_9/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:

save/Read_10/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_20Identitysave/Read_10/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
_output_shapes
:*
T0

save/Read_11/ReadVariableOpReadVariableOp)linear/linear_model/gender/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_22Identitysave/Read_11/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
T0*
_output_shapes

:

save/Read_12/ReadVariableOpReadVariableOp0linear/linear_model/gender_X_type/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes
:	N
r
save/Identity_24Identitysave/Read_12/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	N
g
save/Identity_25Identitysave/Identity_24"/device:CPU:0*
T0*
_output_shapes
:	N

save/Read_13/ReadVariableOpReadVariableOp(linear/linear_model/hJump/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_26Identitysave/Read_13/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_27Identitysave/Identity_26"/device:CPU:0*
_output_shapes

:*
T0

save/Read_14/ReadVariableOpReadVariableOp)linear/linear_model/height/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_28Identitysave/Read_14/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_29Identitysave/Identity_28"/device:CPU:0*
T0*
_output_shapes

:

save/Read_15/ReadVariableOpReadVariableOp(linear/linear_model/tJump/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_30Identitysave/Read_15/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_31Identitysave/Identity_30"/device:CPU:0*
_output_shapes

:*
T0

save/Read_16/ReadVariableOpReadVariableOp'linear/linear_model/type/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_32Identitysave/Read_16/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_33Identitysave/Identity_32"/device:CPU:0*
_output_shapes

:*
T0

save/Read_17/ReadVariableOpReadVariableOp)linear/linear_model/weight/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_34Identitysave/Read_17/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_35Identitysave/Identity_34"/device:CPU:0*
T0*
_output_shapes

:
Р
save/SaveV2_1/tensor_namesConst"/device:CPU:0*т
valueиBе	B*linear/linear_model/age_bucketized/weightsB linear/linear_model/bias_weightsB"linear/linear_model/gender/weightsB)linear/linear_model/gender_X_type/weightsB!linear/linear_model/hJump/weightsB"linear/linear_model/height/weightsB!linear/linear_model/tJump/weightsB linear/linear_model/type/weightsB"linear/linear_model/weight/weights*
dtype0*
_output_shapes
:	
№
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB	B11 1 0,11:0,1B1 0,1B2 1 0,2:0,1B10000 1 0,10000:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B6 1 0,6:0,1B1 1 0,1:0,1*
dtype0*
_output_shapes
:	
Е
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_19save/Identity_21save/Identity_23save/Identity_25save/Identity_27save/Identity_29save/Identity_31save/Identity_33save/Identity_35"/device:CPU:0*
dtypes
2	
Ј
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
д
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
N*
_output_shapes
:*
T0
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
Ј
save/Identity_36Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
э
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueBBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/hiddenlayer_3/biasBdnn/hiddenlayer_3/kernelBdnn/logits/biasBdnn/logits/kernelBglobal_step
ќ
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueBB	100 0,100B8 100 0,8:0,100B70 0,70B100 70 0,100:0,70B48 0,48B70 48 0,70:0,48B34 0,34B48 34 0,48:0,34B1 0,1B34 1 0,34:0,1B 
љ
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*h
_output_shapesV
T:d:d:F:dF:0:F0:":0"::":*
dtypes
2	

save/AssignAssigndnn/hiddenlayer_0/bias/part_0save/RestoreV2*
_output_shapes
:d*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
Ї
save/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save/RestoreV2:1*
_output_shapes

:d*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

save/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save/RestoreV2:2*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:F
Ї
save/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save/RestoreV2:3*
_output_shapes

:dF*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0

save/Assign_4Assigndnn/hiddenlayer_2/bias/part_0save/RestoreV2:4*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
:0
Ї
save/Assign_5Assigndnn/hiddenlayer_2/kernel/part_0save/RestoreV2:5*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:F0

save/Assign_6Assigndnn/hiddenlayer_3/bias/part_0save/RestoreV2:6*
_output_shapes
:"*
T0*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0
Ї
save/Assign_7Assigndnn/hiddenlayer_3/kernel/part_0save/RestoreV2:7*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
_output_shapes

:0"

save/Assign_8Assigndnn/logits/bias/part_0save/RestoreV2:8*
_output_shapes
:*
T0*)
_class
loc:@dnn/logits/bias/part_0

save/Assign_9Assigndnn/logits/kernel/part_0save/RestoreV2:9*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:"
y
save/Assign_10Assignglobal_stepsave/RestoreV2:10*
T0	*
_class
loc:@global_step*
_output_shapes
: 
Щ
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
У
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*т
valueиBе	B*linear/linear_model/age_bucketized/weightsB linear/linear_model/bias_weightsB"linear/linear_model/gender/weightsB)linear/linear_model/gender_X_type/weightsB!linear/linear_model/hJump/weightsB"linear/linear_model/height/weightsB!linear/linear_model/tJump/weightsB linear/linear_model/type/weightsB"linear/linear_model/weight/weights*
dtype0*
_output_shapes
:	
ѓ
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB	B11 1 0,11:0,1B1 0,1B2 1 0,2:0,1B10000 1 0,10000:0,1B1 1 0,1:0,1B1 1 0,1:0,1B1 1 0,1:0,1B6 1 0,6:0,1B1 1 0,1:0,1*
dtype0*
_output_shapes
:	

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes
2	*k
_output_shapesY
W::::	N:::::
f
save/Identity_37Identitysave/RestoreV2_1"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOpAssignVariableOp1linear/linear_model/age_bucketized/weights/part_0save/Identity_37"/device:CPU:0*
dtype0
d
save/Identity_38Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes
:

save/AssignVariableOp_1AssignVariableOp'linear/linear_model/bias_weights/part_0save/Identity_38"/device:CPU:0*
dtype0
h
save/Identity_39Identitysave/RestoreV2_1:2"/device:CPU:0*
_output_shapes

:*
T0

save/AssignVariableOp_2AssignVariableOp)linear/linear_model/gender/weights/part_0save/Identity_39"/device:CPU:0*
dtype0
i
save/Identity_40Identitysave/RestoreV2_1:3"/device:CPU:0*
_output_shapes
:	N*
T0

save/AssignVariableOp_3AssignVariableOp0linear/linear_model/gender_X_type/weights/part_0save/Identity_40"/device:CPU:0*
dtype0
h
save/Identity_41Identitysave/RestoreV2_1:4"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_4AssignVariableOp(linear/linear_model/hJump/weights/part_0save/Identity_41"/device:CPU:0*
dtype0
h
save/Identity_42Identitysave/RestoreV2_1:5"/device:CPU:0*
_output_shapes

:*
T0

save/AssignVariableOp_5AssignVariableOp)linear/linear_model/height/weights/part_0save/Identity_42"/device:CPU:0*
dtype0
h
save/Identity_43Identitysave/RestoreV2_1:6"/device:CPU:0*
_output_shapes

:*
T0

save/AssignVariableOp_6AssignVariableOp(linear/linear_model/tJump/weights/part_0save/Identity_43"/device:CPU:0*
dtype0
h
save/Identity_44Identitysave/RestoreV2_1:7"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_7AssignVariableOp'linear/linear_model/type/weights/part_0save/Identity_44"/device:CPU:0*
dtype0
h
save/Identity_45Identitysave/RestoreV2_1:8"/device:CPU:0*
T0*
_output_shapes

:

save/AssignVariableOp_8AssignVariableOp)linear/linear_model/weight/weights/part_0save/Identity_45"/device:CPU:0*
dtype0

save/restore_shard_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"?
save/Const:0save/Identity_36:0save/restore_all (5 @F8"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"
model_variables
Н
3linear/linear_model/age_bucketized/weights/part_0:08linear/linear_model/age_bucketized/weights/part_0/AssignGlinear/linear_model/age_bucketized/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/age_bucketized/weights  "(2Elinear/linear_model/age_bucketized/weights/part_0/Initializer/zeros:08

+linear/linear_model/gender/weights/part_0:00linear/linear_model/gender/weights/part_0/Assign?linear/linear_model/gender/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/gender/weights  "(2=linear/linear_model/gender/weights/part_0/Initializer/zeros:08
К
2linear/linear_model/gender_X_type/weights/part_0:07linear/linear_model/gender_X_type/weights/part_0/AssignFlinear/linear_model/gender_X_type/weights/part_0/Read/ReadVariableOp:0"9
)linear/linear_model/gender_X_type/weightsN  "N(2Dlinear/linear_model/gender_X_type/weights/part_0/Initializer/zeros:08

*linear/linear_model/hJump/weights/part_0:0/linear/linear_model/hJump/weights/part_0/Assign>linear/linear_model/hJump/weights/part_0/Read/ReadVariableOp:0"/
!linear/linear_model/hJump/weights  "(2<linear/linear_model/hJump/weights/part_0/Initializer/zeros:08

+linear/linear_model/height/weights/part_0:00linear/linear_model/height/weights/part_0/Assign?linear/linear_model/height/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/height/weights  "(2=linear/linear_model/height/weights/part_0/Initializer/zeros:08

*linear/linear_model/tJump/weights/part_0:0/linear/linear_model/tJump/weights/part_0/Assign>linear/linear_model/tJump/weights/part_0/Read/ReadVariableOp:0"/
!linear/linear_model/tJump/weights  "(2<linear/linear_model/tJump/weights/part_0/Initializer/zeros:08

)linear/linear_model/type/weights/part_0:0.linear/linear_model/type/weights/part_0/Assign=linear/linear_model/type/weights/part_0/Read/ReadVariableOp:0".
 linear/linear_model/type/weights  "(2;linear/linear_model/type/weights/part_0/Initializer/zeros:08

+linear/linear_model/weight/weights/part_0:00linear/linear_model/weight/weights/part_0/Assign?linear/linear_model/weight/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/weight/weights  "(2=linear/linear_model/weight/weights/part_0/Initializer/zeros:08

)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"Ѓ
	summaries

/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
/dnn/dnn/hiddenlayer_2/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_2/activation:0
/dnn/dnn/hiddenlayer_3/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_3/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0
linear/bias:0
!linear/fraction_of_zero_weights:0
'linear/linear/fraction_of_zero_values:0
linear/linear/activation:0"$
trainable_variablesџ#ќ#
л
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"&
dnn/hiddenlayer_0/kerneld  "d2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
Х
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"!
dnn/hiddenlayer_0/biasd "d21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
л
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"&
dnn/hiddenlayer_1/kerneldF  "dF2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
Х
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/biasF "F21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
л
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign&dnn/hiddenlayer_2/kernel/part_0/read:0"&
dnn/hiddenlayer_2/kernelF0  "F02<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:08
Х
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign$dnn/hiddenlayer_2/bias/part_0/read:0"!
dnn/hiddenlayer_2/bias0 "021dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:08
л
!dnn/hiddenlayer_3/kernel/part_0:0&dnn/hiddenlayer_3/kernel/part_0/Assign&dnn/hiddenlayer_3/kernel/part_0/read:0"&
dnn/hiddenlayer_3/kernel0"  "0"2<dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform:08
Х
dnn/hiddenlayer_3/bias/part_0:0$dnn/hiddenlayer_3/bias/part_0/Assign$dnn/hiddenlayer_3/bias/part_0/read:0"!
dnn/hiddenlayer_3/bias" ""21dnn/hiddenlayer_3/bias/part_0/Initializer/zeros:08
И
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel"  ""25dnn/logits/kernel/part_0/Initializer/random_uniform:08
Ђ
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:08
Н
3linear/linear_model/age_bucketized/weights/part_0:08linear/linear_model/age_bucketized/weights/part_0/AssignGlinear/linear_model/age_bucketized/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/age_bucketized/weights  "(2Elinear/linear_model/age_bucketized/weights/part_0/Initializer/zeros:08

+linear/linear_model/gender/weights/part_0:00linear/linear_model/gender/weights/part_0/Assign?linear/linear_model/gender/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/gender/weights  "(2=linear/linear_model/gender/weights/part_0/Initializer/zeros:08
К
2linear/linear_model/gender_X_type/weights/part_0:07linear/linear_model/gender_X_type/weights/part_0/AssignFlinear/linear_model/gender_X_type/weights/part_0/Read/ReadVariableOp:0"9
)linear/linear_model/gender_X_type/weightsN  "N(2Dlinear/linear_model/gender_X_type/weights/part_0/Initializer/zeros:08

*linear/linear_model/hJump/weights/part_0:0/linear/linear_model/hJump/weights/part_0/Assign>linear/linear_model/hJump/weights/part_0/Read/ReadVariableOp:0"/
!linear/linear_model/hJump/weights  "(2<linear/linear_model/hJump/weights/part_0/Initializer/zeros:08

+linear/linear_model/height/weights/part_0:00linear/linear_model/height/weights/part_0/Assign?linear/linear_model/height/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/height/weights  "(2=linear/linear_model/height/weights/part_0/Initializer/zeros:08

*linear/linear_model/tJump/weights/part_0:0/linear/linear_model/tJump/weights/part_0/Assign>linear/linear_model/tJump/weights/part_0/Read/ReadVariableOp:0"/
!linear/linear_model/tJump/weights  "(2<linear/linear_model/tJump/weights/part_0/Initializer/zeros:08

)linear/linear_model/type/weights/part_0:0.linear/linear_model/type/weights/part_0/Assign=linear/linear_model/type/weights/part_0/Read/ReadVariableOp:0".
 linear/linear_model/type/weights  "(2;linear/linear_model/type/weights/part_0/Initializer/zeros:08

+linear/linear_model/weight/weights/part_0:00linear/linear_model/weight/weights/part_0/Assign?linear/linear_model/weight/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/weight/weights  "(2=linear/linear_model/weight/weights/part_0/Initializer/zeros:08

)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"ч$
	variablesй$ж$
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
л
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"&
dnn/hiddenlayer_0/kerneld  "d2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
Х
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"!
dnn/hiddenlayer_0/biasd "d21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
л
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"&
dnn/hiddenlayer_1/kerneldF  "dF2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
Х
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/biasF "F21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
л
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign&dnn/hiddenlayer_2/kernel/part_0/read:0"&
dnn/hiddenlayer_2/kernelF0  "F02<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:08
Х
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign$dnn/hiddenlayer_2/bias/part_0/read:0"!
dnn/hiddenlayer_2/bias0 "021dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:08
л
!dnn/hiddenlayer_3/kernel/part_0:0&dnn/hiddenlayer_3/kernel/part_0/Assign&dnn/hiddenlayer_3/kernel/part_0/read:0"&
dnn/hiddenlayer_3/kernel0"  "0"2<dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform:08
Х
dnn/hiddenlayer_3/bias/part_0:0$dnn/hiddenlayer_3/bias/part_0/Assign$dnn/hiddenlayer_3/bias/part_0/read:0"!
dnn/hiddenlayer_3/bias" ""21dnn/hiddenlayer_3/bias/part_0/Initializer/zeros:08
И
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel"  ""25dnn/logits/kernel/part_0/Initializer/random_uniform:08
Ђ
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:08
Н
3linear/linear_model/age_bucketized/weights/part_0:08linear/linear_model/age_bucketized/weights/part_0/AssignGlinear/linear_model/age_bucketized/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/age_bucketized/weights  "(2Elinear/linear_model/age_bucketized/weights/part_0/Initializer/zeros:08

+linear/linear_model/gender/weights/part_0:00linear/linear_model/gender/weights/part_0/Assign?linear/linear_model/gender/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/gender/weights  "(2=linear/linear_model/gender/weights/part_0/Initializer/zeros:08
К
2linear/linear_model/gender_X_type/weights/part_0:07linear/linear_model/gender_X_type/weights/part_0/AssignFlinear/linear_model/gender_X_type/weights/part_0/Read/ReadVariableOp:0"9
)linear/linear_model/gender_X_type/weightsN  "N(2Dlinear/linear_model/gender_X_type/weights/part_0/Initializer/zeros:08

*linear/linear_model/hJump/weights/part_0:0/linear/linear_model/hJump/weights/part_0/Assign>linear/linear_model/hJump/weights/part_0/Read/ReadVariableOp:0"/
!linear/linear_model/hJump/weights  "(2<linear/linear_model/hJump/weights/part_0/Initializer/zeros:08

+linear/linear_model/height/weights/part_0:00linear/linear_model/height/weights/part_0/Assign?linear/linear_model/height/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/height/weights  "(2=linear/linear_model/height/weights/part_0/Initializer/zeros:08

*linear/linear_model/tJump/weights/part_0:0/linear/linear_model/tJump/weights/part_0/Assign>linear/linear_model/tJump/weights/part_0/Read/ReadVariableOp:0"/
!linear/linear_model/tJump/weights  "(2<linear/linear_model/tJump/weights/part_0/Initializer/zeros:08

)linear/linear_model/type/weights/part_0:0.linear/linear_model/type/weights/part_0/Assign=linear/linear_model/type/weights/part_0/Read/ReadVariableOp:0".
 linear/linear_model/type/weights  "(2;linear/linear_model/type/weights/part_0/Initializer/zeros:08

+linear/linear_model/weight/weights/part_0:00linear/linear_model/weight/weights/part_0/Assign?linear/linear_model/weight/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/weight/weights  "(2=linear/linear_model/weight/weights/part_0/Initializer/zeros:08

)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08" 
legacy_init_op


group_deps"ё
table_initializerл
и
_dnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/table_init
[dnn/input_from_feature_columns/input_layer/type_indicator/type_lookup/hash_table/table_init
Mlinear/linear_model_1/linear_model/gender/gender_lookup/hash_table/table_init
Ilinear/linear_model_1/linear_model/type/type_lookup/hash_table/table_init*
predict
)
age"
Placeholder_2:0џџџџџџџџџ
+
tJump"
Placeholder_5:0џџџџџџџџџ
+
hJump"
Placeholder_6:0џџџџџџџџџ
,
weight"
Placeholder_3:0џџџџџџџџџ
*
type"
Placeholder_1:0џџџџџџџџџ
*
gender 
Placeholder:0џџџџџџџџџ
,
height"
Placeholder_4:0џџџџџџџџџ>
logistic2
head/predictions/logistic:0џџџџџџџџџA
	class_ids4
head/predictions/ExpandDims:0	џџџџџџџџџH
probabilities7
 head/predictions/probabilities:0џџџџџџџџџ@
classes5
head/predictions/str_classes:0џџџџџџџџџ&
logits
add:0џџџџџџџџџtensorflow/serving/predict