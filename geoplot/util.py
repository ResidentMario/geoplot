"""
This module contains the static utility methods used internally by the geoplot.core module.
"""

import random
import string
import IPython.core.display


def draw(javascript, css=None):
    """
    Draws a JavaScript object inside of the Jupyter notebook. This is the actual display method that all of the
    geoplots are passed to.

    :param javascript: The JavaScript to be displayed.
    :param css: A CSS stylesheet (optional). Note that CSS will affect all elements in the notebook, so care must be
    taken in style definitions that might be relevant across multiple plots!
    :return: The rendered display object.
    """
    JS_text = string.Template('''
<div id="maindiv${seed}"></div>

<style>

$css

</style>

<script src="http://d3js.org/d3.v3.min.js"></script>

<script>
  $javascript
</script>
''')
    seed = int(random.uniform(0,9999999999))
    JS_text = JS_text.safe_substitute({'seed': seed, 'javascript': javascript, 'css': css})
    JS_text = JS_text.replace('${frame}', '#maindiv{0}'.format(seed))
    return IPython.core.display.display(IPython.core.display.HTML(JS_text))