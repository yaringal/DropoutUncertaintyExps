language: PYTHON
name:     "experiment_BO"

variable {
 name: "tau"
 type: FLOAT
 size: 1
 min:  0.01
 max:  10
}

# variable {
#  name: "alpha_init"
#  type: FLOAT
#  size: 1
#  min:  1e-4
#  max:  1e-0
# }

# Integer example
#
# variable {
#  name: "Y"
#  type: INT
#  size: 5
#  min:  -5
#  max:  5
# }

# Enumeration example
# 
# variable {
#  name: "Z"
#  type: ENUM
#  size: 3
#  options: "foo"
#  options: "bar"
#  options: "baz"
# }


