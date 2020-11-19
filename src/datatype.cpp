//
// Created by serizba on 12/7/20.
//
#include "cppflow/datatype.h"

namespace cppflow {

    /**
     * @return A string representing dt
     *
     */
    std::string to_string(datatype dt) {
        switch (dt) {
            case TF_FLOAT:
                return "TF_FLOAT";
            case TF_DOUBLE:
                return "TF_DOUBLE";
            case TF_INT32:
                return "TF_INT32";
            case TF_UINT8:
                return "TF_UINT8";
            case TF_INT16:
                return "TF_INT16";
            case TF_INT8:
                return "TF_INT8";
            case TF_STRING:
                return "TF_STRING";
            case TF_COMPLEX64:
                return "TF_COMPLEX64";
            case TF_INT64:
                return "TF_INT64";
            case TF_BOOL:
                return "TF_BOOL";
            case TF_QINT8:
                return "TF_QINT8";
            case TF_QUINT8:
                return "TF_QUINT8";
            case TF_QINT32:
                return "TF_QINT32";
            case TF_BFLOAT16:
                return "TF_BFLOAT16";
            case TF_QINT16:
                return "TF_QINT16";
            case TF_QUINT16:
                return "TF_QUINT16";
            case TF_UINT16:
                return "TF_UINT16";
            case TF_COMPLEX128:
                return "TF_COMPLEX128";
            case TF_HALF:
                return "TF_HALF";
            case TF_RESOURCE:
                return "TF_RESOURCE";
            case TF_VARIANT:
                return "TF_VARIANT";
            case TF_UINT32:
                return "TF_UINT32";
            case TF_UINT64:
                return "TF_UINT64";
            default:
                return "DATATYPE_NOT_KNOWN";
        }
    }

    /**
     * @return  The stream os after inserting the string representation of dt
     *
     */
    std::ostream& operator<<(std::ostream& os, datatype dt) {
        os << to_string(dt);
        return os;
    }

}
