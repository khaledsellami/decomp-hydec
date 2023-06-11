import matplotlib.pyplot as plt

from visualization.datapoint import DataPoint

DEFAULT_SIZE = 10


def get_width(text, fontsize):
    """ returns the width in pixels for a given text and fontsize"""
    f = plt.figure()
    r = f.canvas.get_renderer()
    t = plt.text(0.5, 0.5, text, fontsize=fontsize)
    bb = t.get_window_extent(renderer=r)
    width = bb.width
    plt.close(f)
    return width


def get_fontsizes(circles, figsize):
    """ Dynamically measures the text size for each circle """
    fontsizes = list()
    for c in circles:
        r = c.r
        text_limit = r*figsize/2
        if c.ex is not None and not "CC_ID" in c.ex["id"]:
            txt = c.ex["id"].replace("kind: ","").replace("{",'').replace("}","").replace('"',"").replace("$","_")
            max_word = ""
            for word in txt.split(" "):
                if len(word)>len(max_word):
                    max_word = word
            txt = max_word.replace("$","_")
            fontsize = DEFAULT_SIZE
            if get_width(txt,fontsize)<text_limit:
                while get_width(txt,fontsize+1)<text_limit:
                    fontsize += 1
            else:
                while get_width(txt,fontsize-1)>=text_limit and fontsize>1:
                    fontsize -= 1
            fontsizes.append(fontsize)
        else:
            fontsizes.append(DEFAULT_SIZE)
    return fontsizes


def plot_packing_circles(circles, fontsizes, figsize=10, title='circle packing figure', save=None):
    """ Plots the circles """
    # Create just a figure and only one subplot
    px = 1/plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(figsize*px,figsize*px))
    # Title
    fontsize = DEFAULT_SIZE
    text_limit = figsize/4
    if get_width(title,fontsize)<text_limit:
        while get_width(title,fontsize+1)<text_limit:
            fontsize += 1
    else:
        while get_width(title,fontsize-1)>=text_limit and fontsize>1:
            fontsize -= 1
    ax.set_title(title, fontsize=fontsize)
    # Remove axes
    ax.axis('off')
    # Find axis boundaries
    lim = max(
        max(
            abs(circle.x) + circle.r,
            abs(circle.y) + circle.r,
        )
        for circle in circles
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    # print circles
    for circle, fontsize in zip(circles,fontsizes):
        x, y, r = circle.x, circle.y, circle.r
        ax.add_patch(plt.Circle((x, y), r, alpha=0.2, linewidth=2))#, color=circle.ex["color"]
        if circle.ex is not None and not "CC_ID" in circle.ex["id"]:
            text = circle.ex["id"].replace("kind: ","").replace("{",'').replace("}","").replace('"',"").replace("$","_")
            text = " ".join([i for i in text.split(" ") if i!=""])
            if len(text.split(" "))>1:
                txt = ax.text(x, y, text, ha='center', va='center', wrap=True, fontsize=fontsize)
                txt._get_wrap_line_width = lambda : r*figsize
            else:
                txt = ax.text(x, y, text, ha='center', va='center', fontsize=fontsize)
    if save is not None:
        plt.savefig(save)


def from_layers(layers):
    data_points = dict()
    for c, i in layers.iloc[0].to_dict().items():
        c = c.split(".")[-1]
        d = DataPoint(c)
        data_points["C0_"+str(i)] = d
    last_layer = list()
    for l in range(1, layers.shape[0]):
        for c, i in layers.iloc[l].to_dict().items():
            dname = "C{}_{}".format(l,i)
            if not dname in data_points:
                d = DataPoint("CC_ID_{}_{}".format(l, i))
                data_points[dname] = d
                if l == layers.shape[0]-1:
                    last_layer.append(d)
            else:
                d = data_points[dname]
            cname = "C{}_{}".format(l-1, layers.iloc[l-1][c])
            d.add_children({data_points[cname]})
            data_points[cname].parent = d
    return last_layer

# interface_only = False
# project = list(projects.keys())[1]
# db = us.open(projects[project]["project_path"])
# class_refs = [c for c in db.ents("class, interface") if c.parent() is not None]
# true_ind = projects[project]["inds"]["true microservices"]
# true_ind = [int(i) for i in true_ind.split(" ")]
# true_microservices = {i:[] for i in np.unique(ind)}
# micro_colors = dict()
# for i, k in zip(np.random.choice(range(len(plt.cm.tab20.colors)),len(true_microservices),replace=False),true_microservices.keys()):
#     micro_colors[k] = plt.cm.tab20.colors[i]
# class_colors = [None]*len(true_ind)
# for p, i in enumerate(true_ind):
#     true_microservices[i].append(p)
#     class_colors[p] = micro_colors[i]
# microservices = {i:[] for i in np.unique(ind)}
# ind = projects[project]["inds"]["true microservices"]
# ind = [int(i) for i in ind.split(" ")]
# ind_name = "true microservices"
# for p, i in enumerate(ind):
#     microservices[i].append(p)
# data = list()
# for micro in microservices:
#     m = dict()
#     if ind_name == "true microservices":
#         p = class_refs[microservices[micro][0]].parent()
#         while not p.longname().endswith(".java"):
#             p = p.parent()
#         p = p.longname()
#         anchor = "\\microservices\\" + ("java-server\\" if project=="Kanban Board demo" else "")
#         m["id"] = p[p.find(anchor)+len(anchor):p.find("\\src\\")]
#     else:
#         m["id"] = "inferred microservice "+str(micro)
#     m["id"] = "CC_ID"+m["id"]
#     m["children"] = list()
#     for c in microservices[micro]:
#         if interface_only:
#             if "Interface" in class_refs[c].kindname():
#                 m["children"].append({"datum":1,"id":class_refs[c].simplename()})
#         else:
#             m["children"].append({"datum":1,"id":class_refs[c].simplename(),"color":class_colors[c]})
#     m["datum"] = max(len(m["children"]),1)
#     m["color"] = micro_colors[micro]
#     data.append(m)
#
# fontsizes = get_fontsizes(circles, 1500)
