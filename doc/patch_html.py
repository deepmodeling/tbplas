#! /usr/bin/env python

import os


def patch_stat(html):
    patch = """
    <script>
        var _hmt = _hmt || [];
        (function() {
        var hm = document.createElement("script");
        hm.src = "https://hm.baidu.com/hm.js?b04150564935fd379566e6a5a255f1d7";
        var s = document.getElementsByTagName("script")[0]; 
        s.parentNode.insertBefore(hm, s);
        })();
    </script>\n\n"""
    nl_head = None
    with open(html, "r") as f:
        content = f.readlines()
        try:
            nl_head = content.index("</head>\n")
        except ValueError:
            pass
    if nl_head is not None:
        content.insert(nl_head, patch)
        with open(html, "w") as f:
            f.writelines(content)


def patch_icp(html):
    icp = """
    <div align=center>
        <br>
            <a href="http://beian.miit.gov.cn/"; target=_blank>鄂ICP备2022004007号</a>
        </br>
    </div>\n\n"""
    with open(html, "r") as f:
        content = f.readlines()
        nl_footer = content.index("</footer>\n")
        content.insert(nl_footer-1, icp)

    with open(html, "w") as f:
        f.writelines(content)


def main():
    html_files = os.popen("find build/html -name '*.html'").readlines()
    for html in html_files:
        patch_stat(html.rstrip("\n"))
        patch_icp(html.rstrip("\n"))


if __name__ == "__main__":
    main()

