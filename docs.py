
import inspect
import linecache

from examples import tests

view = '''
<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    <title>api example code: </title>
    <link rel="stylesheet" href="pygments.css" type="text/css" />
  </head>
  <body>
    %s
  </body>
</html>
'''


test_list = [
    tests.simple1,
    tests.simple2,
    tests.formatted, 
    tests.group,
    tests.braid,
    tests.gdp_indv,
    tests.gdp_group,
    tests.skills,
]



style1 = '''
.desc{
    font-family:"Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size:14px;fill:#999
}

pre {
    font-family: 'Consolas', 'Deja Vu Sans Mono', 'Bitstream Vera Sans Mono', monospace;
    font-size: 0.95em;
    letter-spacing: 0.015em;
    padding: 0.5em;
    border: 1px solid #ccc;
    background-color: #f8f8f8;
}
'''

def make_doc(style='md'):
    linecache.clearcache()

    samps_blocks = [run_test(func) for func in test_list]
    
    if style is 'htm':
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import HtmlFormatter
        import markdown2

        samps_block_htm = ''.join(htm_example(d) for d in samps_blocks)
        html = view%(samps_block_htm,)
        css = HtmlFormatter().get_style_defs('.highlight')
        with open('t.htm', 'w') as f:
            f.write(html)
        with open('pygments.css', 'w') as f:
            f.write('\n'.join([style1, css]))
        
    if style is 'md':
        with open('README.md', 'w') as f:
            f.write(''.join(md_example(d) for d in samps_blocks))
    
        
        
def run_test(func):
    tests.plt.cla()
    tests.plt.clf()
    ret = func() # run the test; should leave a plot in matplotlib
    #__import__('examples', [func,])
    img_fn = 'public/img/' + func.__name__ + '.png'
    print img_fn
    tests.plt.savefig(img_fn, dpi=80)
    if ret is None: 
        ret = ''
    else:
        ret = str(ret)
    desc = func.__doc__
    code = inspect.getsource(func).replace(desc, '') + ret
    
    d = dict(desc=desc, code=code, fn=img_fn)
    return d

        
def md_example(d):
    md = '''
{desc}
    
```python
{code}
```
<img src="{fn}"/>
'''.format(**d)
    return md
        
        
def htm_example(d):
    """
    """
    markdown = markdown2.Markdown()
    d['desc'] = fmt_desc = markdown.convert(d['desc'])
    d['code'] = highlight(d['code'] , PythonLexer(), HtmlFormatter())
    
    htm = '''
    <div class="desc">{desc}</div>
    <div class="code">{code}</div>
    <img src="{fn}"/>'''.format(**d)

    return htm



if __name__=='__main__':
    make_doc('md')