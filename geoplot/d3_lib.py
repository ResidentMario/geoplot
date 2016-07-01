"""
Glue script. The `draw_graph` method can be used to insert arbitrary JavaScript (in this case, d3) into a
Jupyter notebook cell. Taken from: https://github.com/stitchfix/d3-jupyter-tutorial/blob/master/d3_lib.py.
"""

import random
import inspect, os
from string import Template


def draw_graph(javascript_code):

    JS_text = Template('''

                <div id='maindiv${divnum}'></div>

                <script>
                    $javascript_code
                </script>
                ''')
    divnum = int(random.uniform(0,9999999999))
    return JS_text.safe_substitute({'divnum': divnum, 'main_text': javascript_code})