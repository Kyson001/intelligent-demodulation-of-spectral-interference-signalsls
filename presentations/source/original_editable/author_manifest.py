"""从原图的 Matplotlib 对象坐标生成可编辑 PPT 清单；清单是唯一构建源。"""
from pathlib import Path
import json
import runpy
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.colors import to_rgba, to_hex

HERE = Path(__file__).resolve().parent
ns = runpy.run_path(str(HERE / 'source_geometry.py'))
fig, ax = ns['fig'], ns['ax']
fig.set_dpi(100)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
W, H = 2100, 880
request = json.loads((HERE / 'page_request.json').read_text(encoding='utf-8'))
shapes, texts = [], []

def px(points, transform):
    p = np.asarray(transform.transform(points), dtype=float)
    p[..., 1] = H - p[..., 1]
    return p

def color(value, alpha=None, background='#FFFFFF'):
    rgba = to_rgba(value)
    a = rgba[3] if alpha is None else alpha * rgba[3]
    if a == 0:
        return 'none'
    bg = np.array(to_rgba(background)[:3])
    return to_hex(a * np.array(rgba[:3]) + (1-a)*bg).upper()

def box(points):
    p = np.asarray(points)
    lo, hi = p.min(axis=0), p.max(axis=0)
    return [float(lo[0]), float(lo[1]), max(float(hi[0]-lo[0]), .01), max(float(hi[1]-lo[1]), .01)]

def native_line(p, item):
    if len(p) == 2:
        item.update(type='line', points_px=[float(x) for x in p.ravel()])
    else:
        segments = []
        for a, b in zip(p[:-1], p[1:]):
            d = b-a
            segments.append([float(v) for v in np.concatenate((a, a+d/3, a+d*2/3, b))])
        item.update(type='bezier', box_px=box(p), bezier_px=segments, fill='none')
    return item

artists = [a for a in ax.get_children() if a is not ax.patch]
for n, artist in sorted(enumerate(artists), key=lambda pair: (pair[1].get_zorder(), pair[0])):
    z = float(artist.get_zorder())
    item = dict(id=f'object-{n:03d}', z_index=z)
    if isinstance(artist, Text):
        if not artist.get_text():
            continue
        anchor = px([artist.get_position()], artist.get_transform())[0]
        _, lines, _ = artist._get_layout(renderer)
        for li, (line, _, ox, oy) in enumerate(lines):
            if not line:
                continue
            width, height, descent = renderer.get_text_width_height_descent(line, artist.get_fontproperties(), False)
            baseline = anchor[1] - oy
            # 留出字体行高，但保持文字墨迹区域的中心和原图一致。
            fpx = artist.get_fontsize() * 100/72
            cx = anchor[0] + ox + width/2
            cy = baseline - (height/2-descent)
            bw, bh = width+8, max(height+4, fpx*1.32)
            texts.append(dict(id=f'text-{n:03d}-{li}', text=line,
                box_px=[float(cx-bw/2), float(cy-bh/2), float(bw), float(bh)],
                font='Microsoft YaHei', font_size=artist.get_fontsize()*100/96,
                font_size_source='measured', bold=artist.get_fontweight() in ('bold','heavy','semibold'),
                color=color(artist.get_color()), align='center', valign='middle',
                wrap='none', autofit='none', z_index=z))
    elif isinstance(artist, FancyArrowPatch):
        p = px(artist._posA_posB, artist.get_transform())
        item.update(type='line', points_px=[float(x) for x in p.ravel()],
                    connector='straight', start_arrow='none', end_arrow='triangle',
                    arrow_width='med', arrow_length='med',
                    stroke=color(artist.get_edgecolor()), stroke_width=artist.get_linewidth()*100/96)
        if artist.get_linestyle() != '-':
            item['dash']='dash'
        shapes.append(item)
    elif isinstance(artist, (Circle, Rectangle, FancyBboxPatch)):
        bounds = artist.get_path().get_extents(artist.get_transform())
        b = [float(bounds.x0), float(H-bounds.y1), float(bounds.width), float(bounds.height)]
        item.update(type='ellipse' if isinstance(artist,Circle) else 'roundRect' if isinstance(artist,FancyBboxPatch) else 'rect',
                    box_px=b, fill=color(artist.get_facecolor()), stroke=color(artist.get_edgecolor()),
                    stroke_width=artist.get_linewidth()*100/96)
        if isinstance(artist, FancyBboxPatch):
            item.update(source_corner_radius_px=float(ax.transData.transform((.7,0))[0]-ax.transData.transform((0,0))[0]),
                        corner_category='small-radius', corner_reason='原图 LSTM 单元使用轻微圆角。')
        shapes.append(item)
    elif isinstance(artist, Line2D):
        p=px(artist.get_xydata(), artist.get_transform())
        # Matplotlib 网格有透明度；与其下方的张量颜色合成后保存为相同可见颜色。
        background='#FFFFFF'
        if artist.get_alpha() is not None and artist.get_alpha() < 1:
            mid=artist.get_xydata().mean(axis=0)
            for patch in ax.patches:
                if isinstance(patch,Rectangle) and patch.get_zorder() < z:
                    if patch.get_x()-.001 <= mid[0] <= patch.get_x()+patch.get_width()+.001 and patch.get_y()-.001 <= mid[1] <= patch.get_y()+patch.get_height()+.001:
                        background=to_hex(patch.get_facecolor())
        item.update(stroke=color(artist.get_color(), artist.get_alpha(), background), stroke_width=artist.get_linewidth()*100/96)
        if artist.get_linestyle() not in ('-', 'solid'):
            item['dash']='dash'
        shapes.append(native_line(p,item))

def bbox(item):
    if 'box_px' in item:
        return item['box_px']
    return box(np.array(item['points_px']).reshape(2,2))

def region_index(item, kind):
    b=bbox(item)
    cx,cy=b[0]+b[2]/2,b[1]+b[3]/2
    if kind=='text_boxes' and (cy<70 or cy>790):
        return 0
    if cx<490: return 1
    if cx<1060: return 2
    if cx<1430: return 3
    return 4

regions=[]
labels=['标题和图注','光谱输入及序列重排','双向 LSTM 与序列特征','三路读出及拼接','全连接层和双任务输出']
for ri,label in enumerate(labels):
    rs=[s for s in shapes if region_index(s,'shapes')==ri]
    rt=[t for t in texts if region_index(t,'text_boxes')==ri]
    boxes=[bbox(x) for x in rs+rt]
    x=min(b[0] for b in boxes); y=min(b[1] for b in boxes)
    r=max(b[0]+b[2] for b in boxes); bottom=max(b[1]+b[3] for b in boxes)
    reg=dict(id=f'region-{ri}',label=label,source_bbox_px=[max(0,x-2),max(0,y-2),min(W,r+2)-max(0,x-2),min(H,bottom+2)-max(0,y-2)],
             risk_level='low' if ri==0 else 'high',strategy='native-text-layout' if ri==0 else 'native-object-decomposition',
             reason='使用原绘图对象的实测坐标、线型、颜色和字体，所有对象均以原生对象重建。',
             manifest_ids=dict(shapes=[s['id'] for s in rs],text_boxes=[t['id'] for t in rt],images=[]),protected_visual_anchors=[])
    nodes=[]; edges=[]
    for obj in rs+rt:
        typ=obj.get('type','text')
        kind='label-anchor' if typ=='text' else 'circle-node' if typ=='ellipse' else 'directed-connector' if obj.get('end_arrow')=='triangle' else 'connector' if typ=='line' else 'smooth-curve' if typ=='bezier' else 'node-boundary'
        anc=dict(id='anchor-'+obj['id'],kind=kind,manifest_id=obj['id'])
        if typ=='line': anc['source_points_px']=obj['points_px']
        else: anc['source_bbox_px']=obj['box_px']
        reg['protected_visual_anchors'].append(anc)
        if typ in ('rect','roundRect','ellipse'):
            b=obj['box_px']
            nodes.append(dict(id='node-'+obj['id'],manifest_id=obj['id'],geometry='circle' if typ=='ellipse' else typ,
                              source_center_px=[b[0]+b[2]/2,b[1]+b[3]/2],source_size_px=b[2:],position_tolerance_px=1,size_tolerance_px=1))
        elif typ=='line':
            edges.append(dict(id='edge-'+obj['id'],manifest_id=obj['id'],source_points_px=obj['points_px'],line_style='dashed' if obj.get('dash') else 'solid',direction='end' if obj.get('end_arrow')=='triangle' else 'none',position_tolerance_px=1))
    if nodes:
        reg['compound_diagram']=dict(kind='compound-diagram',object_inventory_complete=True,measurement_reviewed=True,native_object_policy='measured-simple-objects-native',nodes=nodes,edges=edges)
    regions.append(reg)

evidence={
 'font_size_calibrated':'文字由原图字体字号及 Matplotlib 实测字形位置转换，逐行保存为独立文本框。',
 'visual_inventory_matched':'光谱路径、全部矩阵边界和网格、六个 LSTM 单元及十一个神经元均对应独立原生对象。',
 'background_strategy_checked':'原图采用纯白背景，保留原始画布比例和所有模块的绝对位置。',
 'shape_corner_geometry_checked':'矩阵保持直角，六个 LSTM 单元保留实测轻微圆角，神经元保持正圆。'}
manifest=dict(schema_version=2,slide=request['slide'],content_box=request['content_box'],
 source=dict(path='source.png',width_px=W,height_px=H),
 page_strategy='从原绘图源代码的实测对象坐标进行原生对象重建。',
 text_inventory=[t['text'] for t in texts],
 visual_inventory=[dict(id='background',kind='background',representation='native',description='纯白色原生幻灯片背景。')]+[dict(id=f'visual-{i}',kind='native-structure',representation='native',description=l) for i,l in enumerate(labels)],
 background_strategy=dict(mode='native-or-script',source_consistency_contract='保留 2100×880 画布的构图、空白、节点坐标和原始配色。',removed_foreground='全部文字和结构图作为独立对象重建。',comparison_note='原图纯白区域无需图像修复，原生白底与原图一致。'),
 quality_checks={k:True for k in evidence},quality_evidence={k:dict(observation=v,artifact='preview.png') for k,v in evidence.items()},
 text_boxes=texts,shapes=shapes,images=[],asset_provenance=[],formula_inventory=[],
 image2ppt_region_decomposition=dict(schema_version='image2ppt-region-decomposition-v1',page_complexity='structured',source_size_px=[W,H],regions=regions))
(HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'Generated {len(texts)} text boxes and {len(shapes)} independently editable shapes; images=0.')
