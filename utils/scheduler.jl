using ResumableFunctions



@resumable function scheduler(initial_value, decay, min_value)
    value = initial_value
    while true
        @yield value
        value = max(value * decay, min_value)
    end
end


function test()
    step_scheduler = scheduler(1.0, 0.9, 0.1)

    for i=1:100

        step = step_scheduler()
        println(i, " ", step)
    end
end