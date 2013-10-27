import numpy as np

from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib

RECURSION_LIMIT = 2

class Stream(object):
    __slots__ = ['values', 'offset', 'streamset', 'label', 'facecolor',
            'edgecolor', 'xs', 'top_ys', 'bot_ys', 'transshape', 'transsize',
            'alpha','sort_values',
    ]
    def __init__(self, values, sort_values=None, offset=0, streamset=None,  
            label='', facecolor=None, edgecolor=None, alpha=0.5):
            
        self.bot_ys = None
        self.top_ys = None
        self.offset = int(offset)
        
        if values is None:
            if streamset:
                values = streamset.vals.sum(0) + streamset.seps.sum(0) + streamset.pad
                o = streamset.offset
                #values[:o] = np.nan
                self.offset = o
                #values = values[o:]
                #for s in streamset.streams:
                #    s.offset -= o
            else:
                values = [1]
                
        self.values = np.asarray(values, dtype=float)

        ix = np.where(np.isnan(self.values)==False)[0]
        self.offset += ix[0]
        
        self.values = self.values[ix[0]:ix[-1]+1]
        self.values[np.isnan(self.values)] = 0.0
        self.sort_values = sort_values
        if sort_values:
            self.sort_values = np.asarray(sort_values, dtype=float)        
            self.sort_values = self.sort_values[ix[0]:ix[-1]+1]
            
        self.label = str(label)
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        #assert isinstance(streamset, StreamSet)
        self.streamset = streamset
        self.alpha = alpha

    def __str__(self):
        attrs = 'label values offset bot_ys top_ys streamset'.split()
        return 'Stream('+', '.join(a+":"+str(getattr(self,a)).replace('\n','') for a in attrs)+')'

    def __repr__(self): 
        return str(self)
        
class StreamSet(object):
    __slots__ = ['streams', 'pad', 'align', 'normalize', 'margin', 'facecolors',
            'edgecolors', 'vals', 'ends', 'seps', 'levels_within', 
            'label_streams', 'bounds', 'offset', 'sort_vals']
    def __init__(self, streams, pad=0.0, margin=0.0, align=0.5, normalize=False,
            facecolors=cm.BuGn, edgecolors='k'):
        self.streams = streams
        self.pad = pad
        self.align = align
        self.normalize = normalize
        self.margin = margin
        self.offset = 0
        self._calc()
        
        self.levels_within = 0
        self.label_streams = False
        self.facecolors = facecolors
        self.edgecolors = edgecolors
        self.bounds = None
        
    def append(self, stream):
        self.streams.append(stream)
        self._calc()
        
    def _calc(self):
        """
        allocate and calculate the main data arrays (vals, ends, and seps)
        for this streamset.  Is run on init.  re-run if you change stream values, 
        padding, or margins
        """
        streams = self.streams
        h = len(streams)
        o = min(stream.offset for stream in streams)
        w = max([len(stream.values) + stream.offset for stream in streams])-o
        self.offset = o
        self.vals = np.zeros((h, w)) # stream thicknesses
        self.sort_vals = np.zeros((h,w))
        #self.vals[:,:] = np.nan
        self.ends = np.zeros((h, 2), dtype=int) # stream endpoints
        self.seps = np.zeros((h, w)) # padding between streams (includes bottom margin)

        # create an array of plot values
        for i,stream in enumerate(streams):
            #values = np.asarray(stream.values)
            start, stop = stream.offset-self.offset, stream.offset-self.offset + len(stream.values)
            self.ends[i,:] = np.asarray([start, stop])
            #values[np.isnan(values)] = 0.0
            self.vals[i,start:stop] = stream.values
            if stream.sort_values:
                self.sort_vals[i,start:stop] = stream.sort_values
            else:
                self.sort_vals[i,start:stop] = stream.values
            self.seps[i,start:stop] = self.pad
                    
    def bbox(self):
        """ return the bounding box of this streamset """
        return (
            np.min(self.ends)+self.offset, np.max(self.ends)+self.offset, 
            self.bounds[0].min(), self.bounds[1].max(),
        )
        
    def __str__(self):
        attrs = 'vals seps bounds offset ends align streams'.split()
        return 'StreamSet('+', '.join(a+":"+str(getattr(self,a)).replace('\n','') for a in attrs)+')'
        
    def __repr__(self): 
        return str(self)
        
def streamgraph(plt, streamset, transsize=0.5, transshape=0.0, cmap=None, 
    xs=None, legend=False):
    """
    main calling function
    creates a matplotlib figure
    """
    #print '\n',streamset
    
    plt.cla()
    fig = plt.gcf()
    if cmap is not None:
        streamset.facecolors = cmap
    if legend is True: legend=0.2
    if legend:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.0-legend, legend])
        gs.update(wspace=.1)
    else:
        gs = gridspec.GridSpec(1, 1)
        
    ax = plt.subplot(gs[0])

    draw_streamset(plt, streamset, bounds=None,
            transsize=transsize, transshape=transshape, recursion_depth=0,
            do_sort=True, do_labels=not legend)
    xmin, xmax, ymin, ymax = streamset.bbox()
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)

    if xs is not None:
        ax.set_xticks(np.arange(len(xs)))
        ax.set_xticks(np.arange(len(xs))+.5, minor=True)
        ax.set_xticklabels(xs, ha='center', minor=True)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='minor', length=0)

    if legend:
        ax2 = plt.subplot(gs[1])
        streamset2 = streamset_legend(streamset)
        draw_streamset(plt, streamset2, bounds=None,
                transsize=0, recursion_depth=0,
                do_sort=False, do_labels=True)
        xmin, xmax, ymin, ymax = streamset2.bbox()
        plt.xlim(xmin=xmin-streamset2.margin, xmax=xmax+streamset2.margin)
        plt.ylim(ymin=ymin, ymax=ymax)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        
        #return streamset2


def streamset_legend(streamset, level=0):
    """
    return a simplified streamset stripped of data but preserving formatting
    and labels to be used as a legend
    """
    if streamset is None: return None
    streams2 = []
    for s in streamset.streams:
        ss2 = streamset_legend(s.streamset, level=level+1)
        v = 1.0 # default height
        if ss2 is not None:
            v = ss2.vals.max(1).sum() + ss2.seps.max(1).sum() + ss2.margin
        
        vals = [v]*(streamset.levels_within+1)
        s2 = Stream(vals, streamset=ss2, offset=level, label=s.label, 
                facecolor=s.facecolor, edgecolor=s.edgecolor)
        streams2.append(s2)
    return StreamSet(streams2, pad=0.2, margin=0.2, align=0.5, 
        facecolors=streamset.facecolors, edgecolors=streamset.edgecolors)


def draw_streamset(plt, streamset, bounds=None, #offset=0,
        transsize=0.5, transshape=0.0, recursion_depth=0,
        do_sort=True, do_labels=True):
    """
    render the streamset
    """
    if recursion_depth >= RECURSION_LIMIT: return
    streamct = len(streamset.streams)
    
    try:
        face_sm = cm.ScalarMappable(cmap=streamset.facecolors)
        face_cs = face_sm.to_rgba(np.linspace(0,1,streamct))
    except:
        face_cs = [streamset.facecolors]*streamct
    
    try:
        edge_sm = cm.ScalarMappable(cmap=streamset.edgecolors)
        edge_cs = edge_sm.to_rgba(np.linspace(0,1,streamct))
    except:
        edge_cs = [streamset.edgecolors]*streamct
        
    ax = plt.gca()
    normalize = streamset.normalize
    top_margin = streamset.margin
    bot_margin = streamset.margin
    vals = streamset.vals

    if do_sort:
        # create a sorted array of values (stream thicknesses)
        i1 = np.argsort(streamset.sort_vals, 0)
    else:
        i1 = np.zeros(vals.shape, int)
        i1.T[:,:] = np.arange(vals.shape[0]-1,-1,-1)

    vals0 = np.zeros(vals.shape)
    seps0 = np.zeros(vals.shape)
    for j in range(i1.shape[1]):
        vals0[:,j] = streamset.vals[i1[:,j],j]
        seps0[:,j] = streamset.seps[i1[:,j],j]
    seps0[0] += bot_margin-streamset.pad
    #streamset.seps[:,:] = seps0[:,:]

    # sorted array of stream locations
    divs0 = np.cumsum(vals0 + seps0, 0)

    # array of orig ordered locations
    for i in range(divs0.shape[1]):
        divs0[:,i] = divs0[np.argsort(i1[:,i]), i]
    divs = divs0

    # create or use streamset bounds
    if bounds is None:
        lower = np.zeros((divs.shape[1],))
        upper = np.zeros((divs.shape[1],))
        if normalize:
            upper[:] = 1.
            lower[:] = 0.
        else:
            upper[:] = np.max(divs) + top_margin
            lower[:] = 0. #- bot_margin
        bounds = (lower, upper)
    else:
        lower, upper = bounds
        
    streamset.bounds = bounds
    
    # align or scale the streams within the bounds
    top = np.max(divs, 0) + top_margin
    bot = np.zeros(top.shape)# tODO remove
    if normalize:
        # scale
        scale = (upper-lower)/(top-bot)
        #vals[:,:] *= scale
        divs[:,:] *= scale
        divs[:,:] += lower
    else:
        # align
        scale = np.ones(top.shape)
        xtra = lower+((upper-lower)-(top-bot))*streamset.align
        divs[:,:] += xtra

    # generate the streams
    for i, stream in enumerate(streamset.streams):
        start, stop = streamset.ends[i]
        stream.xs = np.arange(start, stop) + streamset.offset
        stream.top_ys = divs[i, start:stop]
        stream.bot_ys = stream.top_ys - vals[i, start:stop]*scale[start:stop]
        if stream.facecolor is None: stream.facecolor = face_cs[i]
        if stream.edgecolor is None: stream.edgecolor = edge_cs[i]
        stream.transshape = transshape
        stream.transsize = transsize
        patch = draw_stream(plt, stream, do_label=do_labels)
        if stream.streamset:
            o = stream.streamset.offset-stream.offset
            draw_streamset(plt, stream.streamset,
                bounds=(stream.bot_ys[o:], stream.top_ys[o:]),
                transsize=transsize,
                transshape=transshape,
                recursion_depth = recursion_depth+1,
                do_sort = do_sort,
                do_labels = do_labels,
            )
            streamset.levels_within = max((
                stream.streamset.levels_within+1, streamset.levels_within))


def draw_stream(plt, stream, do_label=True):
    """
    render a stream
    """
    # TODO (performance) preallocate arrays of verts instead of appending to a list 
    ss = stream.transshape
    ct = len(stream.xs)
    ax = plt.gca()
    trans = [
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO, 
    ]
    halfvs1 = stream.transsize*0.5
    halfvs2 = 1.0-halfvs1
    codes = [Path.MOVETO, Path.LINETO,]
    x0 = stream.xs[0]
    y0 = stream.top_ys[0]
    verts = [(x0+halfvs1, y0), (x0+halfvs2, y0),]
    x = x0

    for j in range(ct)[1:]:
        x = stream.xs[j]
        y1 = stream.top_ys[j-1]
        y2 = stream.top_ys[j]
        codes.extend(trans)
        vs = [(x+ss,y1),(x-ss,y2), (x+halfvs1,y2), (x+halfvs2,y2),]
        verts.extend(vs)

    codes.extend([Path.LINETO,Path.LINETO,])
    verts.extend([(x+halfvs2,stream.bot_ys[-1]),(x+halfvs1,stream.bot_ys[-1])])

    for j in reversed(range(ct)[:-1]):
        x = stream.xs[j]
        y1 = stream.bot_ys[j+1]
        y2 = stream.bot_ys[j]
        codes.extend(trans)
        vs = [(x+1-ss,y1),(x+1+ss,y2), (x+halfvs2,y2), (x+halfvs1,y2),]
        verts.extend(vs)
    codes.extend([Path.LINETO,])
    verts.extend([(x+halfvs1, stream.top_ys[0]),])
    #print len(verts), len(codes), ct*4+4*(ct-1)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, 
        facecolor=stream.facecolor, edgecolor=stream.edgecolor,
        lw=1, alpha=stream.alpha)
    ax.add_patch(patch)

    if do_label:
        top = stream.top_ys[0]
        bot = stream.bot_ys[0]
        mid = (top+bot)/2
        ax.text(x0+halfvs1+0.1, mid, stream.label,
            horizontalalignment='left',
            verticalalignment='center'
            #transform=ax.transAxes
        )
    return patch


