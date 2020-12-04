import tensorflow as tf
from tensorflow.core.framework import op_def_pb2
from google.protobuf import text_format
from termcolor import colored
import re
import textwrap

ops = op_def_pb2.OpList()
text_format.Merge(open('ops.pbtxt').read(), ops)


class Attribute:
    def __init__(self, attr, number_attr_list):

        self.attr = attr
        self.name = self.attr.name


        if self.attr.type == "func": raise Exception("Passing functions as arguments is not yet supported")

        # List attributes are defined as 'list(attr)''
        self.type, self.islist = (self.attr.type, False) if self.attr.type[:4] != 'list' else (self.attr.type[5:-1], True)

        self.number_attr = [i for n, i in number_attr_list if self.name == n]
        self.number_attr, self.type = (self.number_attr[0].name, 'n_attr') if len(self.number_attr) else (None, self.type)

        self.default = bool(len(self.attr.default_value.ListFields())) and not self.islist and self.type not in ['shape', 'tensor']

    def declaration(self):

        # Basic T types attributes are not used
        if self.name == 'T': return ''

        # Number attributes are infered from others (no need for an argument)
        if self.number_attr is not None: return ''

        # Convert from TF types to C++ types
        cpptype = {
            'shape' : 'const std::vector<int64_t>&',
            'int'   : 'int64_t',
            'float' : 'float',
            'string': 'const std::string&',
            'type'  : 'datatype', # Refers to cppflow::datatype
            'bool'  : 'bool',
            'tensor': 'const tensor&'
        }[self.type]

        # Warp list attributes in a C++ vector
        if self.islist:
            cpptype = cpptype.replace('&', '') # Not inner reference types
            cpptype = 'const std::vector<{}>&'.format(cpptype.replace('const', ''))


        # Get the default value for the attribute
        # Not yet supported for lists
        # Not supported for tensors or shape
        if self.default and not self.islist and self.type not in ['shape', 'tensor']:   
            cppdefault = '=' + {
                'int'    : str(self.attr.default_value.i),
                'bool'   : str(self.attr.default_value.b).lower(),
                'string' : '"' + str(self.attr.default_value.s)[2:-1] + '"',
                'float'  : '{:.4e}'.format(self.attr.default_value.f).replace('inf', 'std::numeric_limits<float>::infinity()'),
                'type'   : 'static_cast<datatype>({})'.format(self.attr.default_value.type)
            }[self.type]
        else:
            cppdefault = ''

        # datatype name=defaultval
        return cpptype + ' ' + self.name.replace('template', 'template_arg') + cppdefault

    def code(self):

        # Basic T types attributes are not used
        if self.name == 'T': return ''

        if self.islist:
            return textwrap.dedent({
                'string' : '''
                            std::vector<std::size_t> {0}_sizes; {0}_sizes.reserve({0}.size());
                            std::transform({0}.begin(), {0}.end(), std::back_inserter({0}_sizes), [](const auto& s) {{ return s.size();}});
                            TFE_OpSetAttrStringList(op.get(), "{orig:}", reinterpret_cast<const void *const *>({0}.data()), {0}_sizes.data(), {0}.size());
                            ''',
                'int'    : 'TFE_OpSetAttrIntList(op.get(), "{orig:}", {0}.data(), {0}.size());',
                'float'  : 'TFE_OpSetAttrFloatList(op.get(), "{orig:}", {0}.data(), {0}.size());',
                'bool'   : 'TFE_OpSetAttrBoolList(op.get(), "{orig:}", std::vector<unsigned char>({0}.begin(), {0}.end()).data(), {0}.size());',
                'type'   : 'TFE_OpSetAttrTypeList(op.get(), "{orig:}", reinterpret_cast<const enum TF_DataType *>({0}.data()), {0}.size());',
                'shape'  : '''
                            std::vector<const int64_t*> {0}_values; {0}_values.reserve({0}.size());
                            std::vector<int> {0}_ndims; {0}_ndims.reserve({0}.size());
                            std::transform({0}.begin(), {0}.end(), std::back_inserter({0}_values), [](const auto& v) {{ return v.data();}});
                            std::transform({0}.begin(), {0}.end(), std::back_inserter({0}_ndims), [](const auto& v) {{ return v.size();}});
                            TFE_OpSetAttrShapeList(op.get(), "{orig:}", {0}_values.data(), {0}_ndims.data(), {0}.size(), context::get_status());
                            status_check(context::get_status());
                            ''',
            }[self.type].format(self.name.replace('template', 'template_arg'), orig=self.name)).replace('\n', '\n    ')

        else:
            return textwrap.dedent({
                'shape' : '''
                          TFE_OpSetAttrShape(op.get(), "{orig:}", {0}.data(), {0}.size(), context::get_status());
                          status_check(context::get_status());
                           ''',
                'int'   : 'TFE_OpSetAttrInt(op.get(), "{orig:}", {0});',
                'float' : 'TFE_OpSetAttrFloat(op.get(), "{orig:}", {0});',
                'string': 'TFE_OpSetAttrString(op.get(), "{orig:}", (void*) {0}.c_str(), {0}.size());',
                'type'  : 'TFE_OpSetAttrType(op.get(), "{orig:}", {0});', 
                'bool'  : 'TFE_OpSetAttrBool(op.get(), "{orig:}", (unsigned char){0});',
                'tensor': '''
                           TFE_OpSetAttrTensor(op.get(), "{orig:}", {0}.get_tensor().get(), context::get_status());
                           status_check(context::get_status());
                           ''',
                'n_attr': 'TFE_OpSetAttrInt(op.get(), "{orig:}", {n_attr:}.size());'

            }[self.type].format(self.name.replace('template', 'template_arg'), orig=self.name, n_attr=self.number_attr)).replace('\n', '\n    ')    






class Operation:


    def __init__(self, op):
        self.op = op

        # More than one output?
        if len(self.op.output_arg) != 1: raise Exception("More than one or no output not yet supported")

        self.inputs = [inp for inp in op.input_arg]

        # Number attributes define the length of an input list
        number_attr = [(i.number_attr, i) for i in self.inputs if len(i.number_attr) > 0]


        # Attributes
        self.attr_list = sorted([Attribute(a, number_attr) for a in self.op.attr], key=lambda a: a.default)


    def code(self):

        # C++ function body
        template = textwrap.dedent('''
        {}
        inline {} {}({}{}) {{

            // Define Op
            std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(TFE_NewOp(context::get_context(), "{}", context::get_status()), &TFE_DeleteOp);
            status_check(context::get_status());
            
            // Required input arguments
            {}

            // Attributes
            {}

            // Execute Op
            int num_outputs_op = 1;
            TFE_TensorHandle* res[1] = {{nullptr}};
            TFE_Execute(op.get(), res, &num_outputs_op, context::get_status());
            status_check(context::get_status());
            return tensor(res[0]);
        }}
        ''')

        # Add single input template
        add_inputs = textwrap.dedent('''
            TFE_OpAddInput(op.get(), {}.tfe_handle.get(), context::get_status());
            status_check(context::get_status());
        ''').replace('\n', '\n    ')

        add_inputs_list = textwrap.dedent('''
            std::vector<TFE_TensorHandle*> {0}_handles; {0}_handles.reserve({0}.size());
            std::transform({0}.begin(), {0}.end(), std::back_inserter({0}_handles), [](const auto& t) {{ return t.tfe_handle.get();}});
            TFE_OpAddInputList(op.get(), {0}_handles.data(), {0}.size(), context::get_status());
            status_check(context::get_status());
        ''').replace('\n', '\n    ')

        # Return type of the function
        out = 'tensor' if len(self.op.output_arg) else 'void'

        # snake_case name of the operation
        snk = re.sub(r'(?<!^)(?=[A-Z])', '_', self.op.name).lower().replace('const', 'const_tensor')

        # Required input arguments
        inp = ', '.join(['const std::vector<tensor>&{}'.format(n.name) if len(n.number_attr) or len(n.type_list_attr) else 
                 'const tensor& {}'.format(n.name.replace('tensor', 'input_tensor')) for i, n in enumerate(self.inputs)])

        # Declaration of attributes
        atr = ', '.join(a.declaration() for a in self.attr_list if len(a.declaration()))
        atr = (', ' + atr) if inp != '' and atr != '' else atr

        # Operation original name
        opn = self.op.name

        # Code for input arguments
        inp_code = '\n    '.join(add_inputs_list.format(n.name) if len(n.number_attr) or len(n.type_list_attr) else
                     add_inputs.format(n.name.replace('tensor', 'input_tensor')) for n in self.inputs)

        # Code for attributes
        atr_code = '\n    '.join(a.code() for a in self.attr_list if len(a.code()))

        return template.format('', out, snk, inp, atr, opn, inp_code, atr_code)



ops_file = textwrap.dedent('''
/**
 * @file ops.h
 * TensorFlow raw_ops mappings
 */

#ifndef CPPFLOW2_RAW_OPS_H
#define CPPFLOW2_RAW_OPS_H

#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>

#include <tensorflow/c/eager/c_api.h>
#include <tensorflow/c/tf_datatype.h>
#include <tensorflow/c/tf_tensor.h>

#include "tensor.h"
#include "datatype.h"

namespace cppflow {{

{}

}} // cppflow

#endif

''')



ops_code = ''

num_ops = 0

# All TF C API operations correspond with tf.raw_ops
for op_name in sorted(dir(tf.raw_ops)):
    if not op_name.startswith("_"):

        num_ops += 1
        #if num_ops == 51:
        #    break

        try:

            # Grab operation definition
            op = [op for op in ops.op if op.name == op_name]
            if len(op) == 0: raise Exception("Operation not found")
            
            op = Operation(op[0])

            ops_code += op.code()


            # Everything was ok!
            print('{:<50}  [{}]'.format(op_name, colored('  Ok  ', 'green')))
        except Exception as err:
            print('{:<50}  [{}]'.format(op_name, colored('Failed', 'red')))
            print('    ', err)


with open('../raw_ops.h', 'w') as f:
    f.write(ops_file.format(ops_code))
