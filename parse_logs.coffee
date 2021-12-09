c = console.log.bind console


#  We  wouldn't want to use nodemon outside of the script.  The file watch function
# will be inside this program, when it detects a change to log, it parses the file
# and outputs results to a different log file.

# In the main.rs file we should have a routine on startup that kills that main log 
# file so that we don't get appends 


fs = require 'fs'
path = require 'path'


watch = require 'node-watch'






file_path = path.join __dirname, 'logs', 'log_main.txt'



stream = fs.createWriteStream 'parsed_logs.txt', {flags: 'a'}


thing_do = ->
    fs.readFile file_path, {encoding: 'utf-8'}, (err, data) ->
        # c data


        # c typeof data


        arr = data.split '\n'
        c arr

        for str in arr
            if str.includes "Validation Error"
                c "#{str} \n"
                # stream.write str + '\n'

                fs.appendFile './logs/parsed_logs.txt', str, (err) ->
                    if err then c err

                fs.appendFile './logs/parsed_logs.txt', "\n \n", (err) ->
                    if err then c err


watch './logs/log_main.txt', {}, (evt, name) ->
    # c '&s changed', name

    thing_do()


        # if data.includes(" VK_NV_representative_fragment_test ")
        #     c data


# stream.end()

thing_do()