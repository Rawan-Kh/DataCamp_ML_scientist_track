<!-- You correctly chose the incorrect statement! Since "/html/body/*" selects all elements one generation below the body element without concern of the tag type, it selects all children of the body element. On the other hand, "/html/body//*" selects all elements from all future generations of the body element (that is, all descendants of the body) regardless of tag type. -->

--------------
# Create an XPath string to direct to children of body element
xpath = "/html/body/*" 

# Print out the number of elements selected
how_many_elements( xpath )
--------
# Create an XPath string to the desired paragraph element
xpath = "/html/body/div[1]/div/p" 

# Print out the element text
print_element_text( xpath )
-----------
# Create an Xpath string to select desired p element
xpath = '//*[@id="div3"]/p'

# Print out selection text
print_element_text( xpath )

-----------
# Create an XPath string to select p element by class
xpath = '//p[@class="class-1 class-2"]'

# Print out select text
print_element_text( xpath )
<!-- Notice how sensitive it is to use an XPath string to select by class -- you even need to get the spacing correct! -->
-----------

# Create an xpath to the href attribute
xpath = '//p[@id="p2"]/a/@href'

# Print out the selection(s); there should be only one
print_attribute( xpath )
<!-- You were able to select the correct paragraph element by id, narrow down to the hyperlink child, and finally direct to the href attribute! -->
---------

# Create an xpath to the href attributes
xpath = '//a[contains(href,"package-snippet")]/@href'

# Print out how many elements are selected
how_many_elements( xpath )
# Preview the selected elements
preview( xpath )
<!-- You have selected all 10 href attributes. -->
----------

# Chain together xpath methods to select desired p element
sel.xpath( '//div' ).xpath( './span/p[3]' )
---------
from scrapy import Selector

# Create a Selector selecting html as the HTML document
sel = Selector( text = html )

# Create a SelectorList of all div elements in the HTML document
divs = sel.xpath( '//div' )

--------
# Import a scrapy Selector
from scrapy import Selector

# Import requests
import requests

# Create the string html containing the HTML source
html = requests.get( url ).content

# Create the Selector object sel from html
sel = Selector( text=html )

# Print out the number of elements in the HTML document
print( "There are 1020 elements in the HTML document.")
print( "You have found: ", len( sel.xpath('//*') ) )

-------

