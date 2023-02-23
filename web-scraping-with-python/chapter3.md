# Create the XPath string equivalent to the CSS Locator 
xpath = '/html/body/span[1]//a'

# Create the CSS Locator string equivalent to the XPath
css_locator = 'html > body > span:nth-of-type(1) a'

---------wrong
# Create the XPath string equivalent to the CSS Locator 
xpath = '// div[@id="uid"] / span // h4 '

# Create the CSS Locator string equivalent to the XPath
css_locator = 'div#uid > span h4'

-------right
# Create the XPath string equivalent to the CSS Locator 
xpath = '//div[@id="uid"]/span//h4'

# Create the CSS Locator string equivalent to the XPath
css_locator = 'div#uid > span h4'

------------
from scrapy import Selector

# Create a selector from the html (of a secret website)
sel = Selector( text = html )

# Fill in the blank
css_locator = 'div.course-block > a'

# Print the number of selected elements.
how_many_elements( css_locator )
<!-- You have found all the hyperlink children of the course-block div elements. -->
-------

# Create the CSS Locator to all children of the element whose id is uid
css_locator = '#uid > * '
<!-- You were able to combine your knowledge of CSS Locators with your instincts for the wildcard character to do the job!! -->
--------

from scrapy import Selector

# Create a selector object from a secret website
sel = Selector( text=html )

# Select all hyperlinks of div elements belonging to class "course-block"
course_as = sel.css( 'div.course-block > a' )

# Selecting all href attributes chaining with css
hrefs_from_css = course_as.css( '::attr(href)' )

# Selecting all href attributes chaining with xpath
hrefs_from_xpath = course_as.xpath('./@href')
You even remembered to use the period as glue when creating the XPath. Maybe you noticed, but if you chain with css (rather than xpath), you don't need any glue!
--------

<!-- Assign to the variable xpath an XPath string directing to the text within the paragraph p element with id equal to p3, which does not include the text of future generations of this p element. -->
# Create an XPath string to the desired text.
xpath = '//p[@id="p3"]/text()'

# Create a CSS Locator string to the desired text.
css_locator = 'p#p3::text'

# Print the text from our selections
print_results( xpath, css_locator )
<!-- You were able to direct to text with both XPath and CSS Locator strings! -->
----------
<!-- Assign to the variable xpath an XPath string directing to the text within the paragraph p element with id equal to p3, which includes the text of future generations of this p element. -->
# Create an XPath string to the desired text.
xpath = '//p[@id="p3"]//text()'

# Create a CSS Locator string to the desired text.
css_locator = 'p#p3 ::text'

# Print the text from our selections
print_results( xpath, css_locator )
---------

# Get the URL to the website loaded in response
this_url = response.url

# Get the title of the website loaded in response
this_title = response.xpath( '/html/head/title/text()' ).extract_first()

# Print out our findings
print_url_title( this_url, this_title )

-------------
