c = console.log.bind console


{ spawn, exec } = require 'child_process'


vert = exec ('glslc.exe ./shaders/s1.vert -o ./spv/s1.vert.spv')
frag = exec ('glslc.exe ./shaders/s1.frag -o ./spv/s1.frag.spv')



# vert.stderr.on 'data', (data) ->
#     c data
# glslc.exe ./shaders/src/shader.vert -o ./shaders/spv/vert.spv
# glslc.exe ./shaders/src/shader.frag -o ./shaders/spv/frag.spv
# pause