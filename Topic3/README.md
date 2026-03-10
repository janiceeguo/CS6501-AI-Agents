# Agent Tool Use

- Jupyter notebook (all code) in `tools.ipynb`
- Separate code for each task are found in each `.py` files
- Separate complete outputs and graphs for each task in the `outputs` folder

## Task 1

|**Task name**|**real time**|**user time**|**system time**|
|---|---|---|---|
|abstract algebra|0m53.963s|0m28.832s|0m14.331s|
|anatomy|0m27.802s|0m18.263s|0m3.707s|
|ollama sequential|0m58.344s|0m8.263s|0m1.193s|
|ollama parallel|0m56.131s|0m8.368s|0m1.110s|

The timing results show that when both tasks were run sequentially through the Ollama server, the total real time was 58 seconds, which is shorter than the sum of the individual runs. The smaller user and system time shows that Ollama offloads these heavier GPU tasks to its server. These times are almost the same between parallel, showing that the server can handle concurrent requests, though the speedup is very minimal for tasks of this size.

## Task 4:

I added a calculator function with ability to add, subtract, multiply, divide, round, sin, cos, tan, find circle area, rectangle area, triangle area, and distance between two points. I also added a function to count the number of letters in a string, and a tool to count the number of words and characters in a piece of text. It was also able to call tools multiple times in one itteration. For my fourth query "In Mississippi riverboats, count i's and s's, compute the sin of their difference, tell me the weather in London, and give text statistics." the LLM was able to combine all the different tools smoothly, but it tended to take shortcuts by combining steps. I had to do a bit of prompt engineering to force the model to separate the tools and steps that it would take to answer my prompt. My prompt that reached the five turn limit was: 

    "You must follow each step in order and use tools whenever applicable. "
    "Step 1: Count i's in Mississippi riverboats. "
    "Step 2: Count the number of characters in Mississippi riverboats. "
    "Step 3: Compute the area of a rectangle that has length = # of i's and width = # of characters. "
    "Step 4: Take the cosine of the rectangle's area. "
    "Step 5: Multiply the cosine value by 10 and round to the nearest integer. "
    "Do not skip steps and do not combine steps."

## Task 5:

I was able to have a conversation where I asked the model to get the character count and calculate the area of a rectangle in a single query (output5_continuous.txt). I was also able to recover conversation by manually causing a keyboard crash right after I entered a query, and when restarting the model it was able to give the correct output to the query before the crash (output5_recovery.txt). Lastly, I was able to see that the model remembers conversation history by asking it to define a variable in a previous query, then asking it to recall the value of that variable several queries later (output5_context.txt).

## Task 6: 

One opportunity for parallelization in the agent is in the `call_tools` node, where tool calls returned by the language model are executed sequentially. When the LLM produces a response containing multiple `tool_calls`, the current implementation iterates through them in a `for` loop and invokes each tool one at a time. However, these tool calls are currently independent of one another. For example, my previous query "In Mississippi riverboats, count i's and s's, compute the sin of their difference, tell me the weather in London, and give text statistics" asks the model to request a weather lookup, a calculator operation, and a text statistics computation in the same response. Since none of these operations rely on the results of the others, they could safely be executed concurrently rather than sequentially. Running them in parallel would allow the agent to reduce overall latency, especially if some tools involve slower operations such as external API requests or heavier computation. By executing all requested tools simultaneously and collecting their results afterward, the system could produce the same final output while making more efficient use of available computational resources.
