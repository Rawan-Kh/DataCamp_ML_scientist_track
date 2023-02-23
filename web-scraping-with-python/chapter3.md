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
# Create a CSS Locator string to the desired hyperlink elements
css_locator = 'a.course-block__link'

# Select the hyperlink elements from response and sel
response_as = response.css(css_locator)
sel_as = sel.css(css_locator)

# Examine similarity
nr = len( response_as )
ns = len( sel_as )
for i in range( min(nr, ns, 2) ):
  print( "Element %d from response: %s" % (i+1, response_as[i]) )
  print( "Element %d from sel: %s" % (i+1, sel_as[i]) )
  print( "" )

<!-- Surely you see now that if we use a response variable to select elements, we are back in the realm of Selectors and SelectorLists. -->
------------
# Select all desired div elements
divs = response.css( 'div.course-block' )

# Take the first div element
first_div = divs[0]

# Extract the text from the h4 element in first_div
h4_text = first_div.css('h4::text').extract_first()

# Print out the text
print( "The text from the h4 element is:", h4_text )
<!-- That's right -- the variable divs was a SelectorList, so the first element in this list is a Selector object. -->
--------------
# Create a SelectorList of the course titles
crs_title_els = response.css( 'h4::text' )

# Extract the course titles 
crs_titles = crs_title_els.extract()

# Print out the course titles 
for el in crs_titles:
  print( ">>", el )

-------
# Calculate the number of children of the mystery element
how_many_kids = len( mystery.xpath( './*' ) )

# Print out the number
print( "The number of elements you selected was:", how_many_kids )

<!-- Now you know the trick! In xpath, you can use the XPath string './*' to direct to the children of the currently selected element! -->
-----------


